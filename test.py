#%%
from cgi import test
from inspect import indentsize
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from seaborn import axisgrid
from sklearn.model_selection import train_test_split
# from torch._C import namedtuple_sign_logabsdet
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform,cdist
import pykrige.variogram_models as vm
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.benchmark = True
# seed=218
# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
print(device)
# %%
input_data = pd.read_csv('./input/useful_data/input_data.csv')
input_data['t']=np.where(input_data['t'].values<=0,0.1,input_data['t'].values)

input_data
#%%
a=input_data.groupby(['x','y'],as_index=False).min()
len(a[a['h_z']<=-1000]),len(a[a['h_z']<=-3000]),len(a[a['h_z']<=-5000])

#%%
input_data = input_data.sort_values(
    'h_z', ascending=False, ignore_index=True)
def extra_split(df):
    train_data,test_data = train_test_split(input_data, test_size=0.1, shuffle=False)
    return train_data,test_data

def boring_split(df):
    xy=df.groupby(['x','y'],as_index=False).mean().loc[:,['x','y']].values
    xy,xy_test=train_test_split(xy,test_size=0.1)
    train_data=pd.DataFrame()
    for x,y in xy:
        train_data_=df[(df['x']==x)&(df['y']==y)]
        train_data=pd.concat([train_data,train_data_],axis=0)
    test_data=pd.DataFrame()
    for x,y in xy_test:
        test_data_=df[(df['x']==x)&(df['y']==y)]
        test_data=pd.concat([test_data,test_data_],axis=0)
    return train_data,test_data

train_data, test_data = extra_split(input_data)

dnn_features=['x','y','h_z']#,'z','curie','near_t','near_dist','variogram'
train_features=train_data[dnn_features]
train_target=train_data['t']
test_features=test_data[dnn_features]
test_target=test_data['t']
est_features=est_data[dnn_features]
# %%
sdscaler = preprocessing.MinMaxScaler()
train_features=sdscaler.fit_transform(train_features)
test_features=sdscaler.transform(test_features)
est_features=sdscaler.transform(est_features)

log_data = np.log(train_target)
log_mean = log_data.mean()
log_std = log_data.std()
omax=600
omin=0

train_feature = Variable(torch.from_numpy(train_features).float()).to(device)
train_target = Variable(torch.from_numpy(train_target.values.reshape(-1,1)).float()).to(device)
test_feature = Variable(torch.from_numpy(test_features).float()).to(device)
test_target = Variable(torch.from_numpy(test_target.values.reshape(-1,1)).float()).to(device)
est_features = Variable(torch.from_numpy(est_features).float()).to(device)
# %%
def convert_output(output_data, debug=False):

    ret = output_data

    global log_std,log_mean

    ret=ret*log_std+log_mean
    
    ret = torch.exp(ret)
    
    if debug:
        print(ret)

    return ret

# 入力値の変換
def convert_data(input_data, debug=False):
    ret = convert_output(model(input_data), debug)
    return ret

def Min_t(output, omin):
    min_t = ((output - omin).clamp(max=0) ** 2).mean()
    return min_t

def Max_t(output, omax):
    max_t = ((output - omax).clamp(min=0) ** 2).mean()
    return max_t

# %%
class Net(nn.Module):
    def __init__(self, input_size, output_size, unit_size):
        super(Net, self).__init__()
        self.fc_s = nn.Linear(input_size, unit_size)
        self.fc2 = nn.Linear(int(unit_size), int(unit_size))
        self.fc3 = nn.Linear(int(unit_size), int(unit_size))
        self.fc_e = nn.Linear(int(unit_size), output_size)
        
    def forward(self, x):
        
        x = F.relu(self.fc_s(x))
        # x=F.dropout(x,p=0.2)
        x = F.relu(self.fc2(x))
        x=F.dropout(x,p=0.2)
        x = F.relu(self.fc3(x))
        # x=F.dropout(x,p=0.2)
        x = self.fc_e(x)
        
        return x
# %%
no=11

prediction_est_list=[]
prediction_train_list=[]
prediction_test_list=[]
loss_list_master=[]
for ave in range(3):
    print(ave,'-'*50)
    loss_list=[]
    model=Net(train_feature.shape[1],1,120).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for epoch in range(2001):
        optimizer.zero_grad()
        prediction_est=convert_data(est_features)
        prediction_train=convert_data(train_feature)
        prediction_test=convert_data(test_feature)

        loss = F.mse_loss(prediction_train, train_target)
        loss_v = F.mse_loss(prediction_test, test_target)   

        loss_sum=loss+Max_t(prediction_est,omax)+Min_t(prediction_est,omin)
        loss_sum.backward() 
        optimizer.step()
        if epoch%100==0:
            print(f'{epoch},{np.sqrt(loss.data.item())},{np.sqrt(loss_v.data.item())}')
            loss_list.append([epoch,np.sqrt(loss.data.item()),np.sqrt(loss_v.data.item())])
    model.eval()
    prediction_est_list.append(convert_data(est_features).detach().cpu().numpy())
    prediction_train_list.append(convert_data(train_feature).detach().cpu().numpy())
    prediction_test_list.append(convert_data(test_feature).detach().cpu().numpy())
    pd.DataFrame(np.array(loss_list),columns=['epoch','train_loss','test_loss']).to_csv(f'./output/learning_curve/lc_{no}_{ave}.csv',index=False)
#%%

import pandas as pd
# %%
a=pd.read_csv('./input/xy_grid_albers.csv')
a=a.astype(float)
a.dtypes
#%%
a=a.groupby(['x','y'],as_index=False).mean()[['x','y']]
a
# %%
a['ido'].unique().shape
# %%
#%%
import pandas as pd
import numpy as np
# %%
def grid_albers(sep_xy,sep_z):#sep_xy(km),sep_z(m)#単位に注意
    #sep_xy(km),sep_z(m)#単位に注意
    
    #df1
    ido=range(950,1151,sep_xy)
    keido=range(2950,3051,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df1=pd.DataFrame(data,columns=['y','x','z'])
    #df2
    ido=range(800,1151,sep_xy)
    keido=range(3050,3251,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df2=pd.DataFrame(data,columns=['y','x','z'])
    #df3
    ido=range(1150,1451,sep_xy)
    keido=range(3100,3251,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df3=pd.DataFrame(data,columns=['y','x','z'])
    #df4
    ido=range(1050,1451,sep_xy)
    keido=range(3250,3351,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df4=pd.DataFrame(data,columns=['y','x','z'])
    #df5
    ido=range(1200,1751,sep_xy)
    keido=range(3350,3451,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df5=pd.DataFrame(data,columns=['y','x','z'])
    #df6
    ido=range(1200,2351,sep_xy)
    keido=range(3450,3551,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df6=pd.DataFrame(data,columns=['y','x','z'])
    #df7
    ido=range(1400,2201,sep_xy)
    keido=range(3550,3651,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df7=pd.DataFrame(data,columns=['y','x','z'])
    #df8
    ido=range(1400,2001,sep_xy)
    keido=range(3650,3751,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df8=pd.DataFrame(data,columns=['y','x','z'])
    #df9
    ido=range(2250,2751,sep_xy)
    keido=range(3300,3451,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df9=pd.DataFrame(data,columns=['y','x','z'])
    #df10
    ido=range(2450,2751,sep_xy)
    keido=range(3450,3651,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df10=pd.DataFrame(data,columns=['y','x','z'])
    #まとめ
    df=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10])
    df=df.loc[:,['x','y','z']]
    df['x']=df['x']-100000
    df['y']=df['y']-100000
    df=df.drop_duplicates(['x','y','z'])
    
    return df
# %%
df=grid_albers(5,5501)
df
# %%
# df.to_csv('./成形前データ/grid_500.csv')
# %%
albers=pd.read_csv('./input/xy_h_grid_albers.csv')
albers
# %%
df=df.merge(albers,on=['x','y'],how='left').dropna()
df
#%%
import datetime
print(datetime.datetime.now())

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
profile_db=pd.read_csv('./input/useful_data/database/profile_db.csv')
geothermal_db=pd.read_csv('./input/useful_data/database/geothermal_db.csv')
gsj_db=pd.read_csv('./input/useful_data/database/GSJ_db.csv')
profile_db['no']=0
geothermal_db['no']=1
gsj_db['no']=2

master=pd.concat([profile_db,geothermal_db,gsj_db]).reset_index(drop=True)
master=master.round()

#%%
master[(master['x']==3440992.0 )&(master['y']==2015840.0)]
#%%
xy_profile_db=profile_db.groupby(['x','y'],as_index=False).mean()[['x','y']]
xy_profile_db[xy_profile_db.round().duplicated(keep=False)]
#%%
xy_geothermal_db=geothermal_db.groupby(['x','y'],as_index=False).mean()[['x','y']]
xy_geothermal_db[xy_geothermal_db.round().duplicated(keep=False)]
#%%
xy_gsj_db=gsj_db.groupby(['x','y'],as_index=False).mean()[['x','y']]
xy_gsj_db[xy_gsj_db.round().duplicated(keep=False)]
# %%
xy_master=master.groupby(['x','y'],as_index=False).mean()[['x','y','no']]
a=xy_master.round()[xy_master.round().duplicated(keep=False,subset=['x','y'])]
a['no'].value_counts()
#%%
a['x'].value_counts().max()
#%%
master[master.duplicated(keep=False,subset=['x','y','h_z'])].sort_values(['x','h_z'])
# %%
profile_db[profile_db.duplicated(keep=False,subset=['x','y','h_z'])].sort_values(['x','h_z'])
# %%
geothermal_db[geothermal_db.duplicated(keep=False,subset=['x','y','h_z'])].sort_values(['x','h_z'])

# %%
gsj_db[gsj_db.duplicated(keep=False,subset=['x','y','h_z'])].sort_values(['x','h_z'])

# %%
master.shape,profile_db.shape,geothermal_db.shape,gsj_db.shape
# %%
for i in [master,profile_db,geothermal_db,gsj_db]:
    plt.plot(i['t'],i['h_z'],'.')
    plt.show()
    print(i.groupby(['x','y'],as_index=False).mean()[['x','y']].shape)
# %%
plt.hist(np.log(master[master['t']>0]['t']),bins=100)
plt.show()
plt.hist(master['t'],bins=100)
plt.show()
# %%
master.describe()
# %%
master['x'],master['y']=master['x'].round(2),master['y'].round(2)
master
# %%
master.groupby(['x','y'],as_index=False).mean()
# %%
a=pd.read_csv('./input/curie_point/add_grid_curie.csv')
a[a['curie']<=6]
# %%
# %%
def extra_split(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.1, shuffle=False)
    return train_data, test_data
input_data = pd.read_csv('./input/useful_data/input_data.csv')
train_data, test_data = extra_split(input_data)
#%%
train_data.h_z.min()
# %%
a=["a","b","c","d"]
b=a.copy()
b+=["e"]
a,b
# %%
input_data = pd.read_csv('./input/useful_data/input_data.csv')
input_data=input_data.merge(pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv'),on=["x","y"],how="left")
input_data
# %%
import seaborn as sns
import matplotlib.pyplot as plt
# %%
plt.rcParams['font.family'] = 'Yu Gothic'
sns.barplot(x='group_ja',y='t',data=input_data[input_data.h_z==-500])
# %%
input_data['group_rank']=input_data['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
input_data
# %%
import pandas as pd
def preprocess_input(df):
    df['t'] = np.where(df['t'].values <= 0, 0.1, df['t'].values)
    add_input_volcano = pd.read_csv('./input/volcano/add_input_volcano.csv')
    add_input_curie = pd.read_csv('./input/curie_point/add_input_curie.csv')
    add_input_tishitsu = pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
    add_input_onsen = pd.read_csv('./input/onsen/add_input_onsen.csv')
    add_input_kmeans = pd.read_csv('./input/k_means/add_input_kmeans.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])
    df = df.merge(add_input_kmeans, how='left', on=['x', 'y'])
    
    df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    df=df.drop(['symbol','symbol_freq','formationAge_ja', 'group_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    df['age']=(df['age_a']+df['age_b'])/2

    return df
input_data = pd.read_csv('./input/useful_data/input_data.csv')
input_data = preprocess_input(input_data)
# %%
input_data.columns
# %%
pd.read_csv('./成形前データ/hyoukou_detail.csv').dropna()
# %%
("all"  in "all2")
# %%
import pandas as pd
# %%
input_data = pd.read_csv('./input/useful_data/input_data.csv')
input_data.h_z=input_data.h_z.round(-1)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
input_data=input_data[input_data.h_z==0][["x","y"]]
input_data.to_csv("./output/arcgis/0.csv")
# %%
wgs_albers=pd.read_csv("./input/WGS_to_albers/input_WGS_albers.csv")
wgs_albers.x,wgs_albers.y=wgs_albers.x.round(),wgs_albers.y.round()
wgs_albers=wgs_albers.groupby(["x","y"],as_index=False).mean()
wgs_albers
# %%
input_data.merge(wgs_albers,on=["x","y"],how="left")[["keido",""]]
# %%
# %%
from numpy.lib.ufunclike import fix
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from optuna.samplers import TPESampler
import optuna
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from xgboost.callback import LearningRateScheduler
import time
import datetime
print(datetime.datetime.now())
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
#%%
# %%
from numpy.lib.ufunclike import fix
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from optuna.samplers import TPESampler
import optuna
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from xgboost.callback import LearningRateScheduler
import time
import datetime
print(datetime.datetime.now())
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
#%%
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(0)

def preprocess_input(df):
    df['t'] = np.where(df['t'].values <= 0, 0.1, df['t'].values)
    add_input_volcano = pd.read_csv('./input/volcano/add_input_volcano.csv')
    add_input_curie = pd.read_csv('./input/curie_point/add_input_curie.csv')
    add_input_tishitsu = pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
    add_input_onsen = pd.read_csv('./input/onsen/add_input_onsen.csv')
    add_input_kmeans = pd.read_csv('./input/k_means/add_input_kmeans.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])
    df = df.merge(add_input_kmeans, how='left', on=['x', 'y'])

    df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    df=df.drop(['symbol','symbol_freq','formationAge_ja', 'group_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    df['age']=(df['age_a']+df['age_b'])/2

    return df

def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('./input/volcano/add_grid_volcano.csv')
    add_grid_curie = pd.read_csv('./input/curie_point/add_grid_curie.csv')
    add_grid_tishitsu = pd.read_csv('./input/tishitsu/add_grid_tishitsu_pred.csv')
    add_grid_onsen = pd.read_csv('./input/onsen/add_grid_onsen.csv')
    add_grid_kmeans = pd.read_csv('./input/k_means/add_grid_kmeans.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])
    df = df.merge(add_grid_kmeans, how='left', on=['x', 'y'])

    df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    df=df.drop(['symbol','symbol_freq','formationAge_ja', 'group_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    df['age']=(df['age_a']+df['age_b'])/2

    return df


def grad_calc(df):
    xy_unique = df.groupby(['x', 'y'], as_index=False).mean()[
        ['x', 'y']].values
    for x, y in xy_unique:
        zt = df[(df['x'] == x) & (df['y'] == y)][['z', 't']]
        if zt.shape[0] == 1:
            zt = pd.DataFrame(np.array([[0, 0]]), columns=[
                              'z', 't']).append(zt)
        grad = np.polyfit(zt['z'], zt['t'], 1)[0]*1000
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad'] = grad
    return df


def grad_maxmin_calc(df):
    xy_unique = df.groupby(['x', 'y'], as_index=False).mean()[
        ['x', 'y']].values
    for x, y in xy_unique:
        zt = df[(df['x'] == x) & (
            df['y'] == y)][['z', 't']]
        if zt.shape[0] == 1:
            zt = pd.DataFrame(np.array([[0, 0]]), columns=[
                              'z', 't']).append(zt)
        zt = zt.sort_values('z')
        z_diff = np.diff(zt['z'])
        t_diff = np.diff(zt['t'])
        grad = (t_diff/z_diff)*1000
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_max'] = max(grad)
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_min'] = min(grad)
    return df


def semi_variogram_feature(df_input, df):

    dict_condition = {
        'hokkaidou': [(df_input['x']-df_input['y']) <= 1200000, (df['x']-df['y']) <= 1200000],
        'higashi': [((df_input['x']-df_input['y']) > 1200000) & ((df_input['x']+df_input['y']) >= 5000000), ((df['x']-df['y']) > 1200000) & ((df['x']+df['y']) >= 5000000)],
        'nishi': [((df_input['y']+(2/3)*df_input['x']) > 3050000) & ((df_input['x']+df_input['y']) < 5000000), ((df['y']+(2/3)*df['x']) > 3050000) & ((df['x']+df['y']) < 5000000)],
        'kyusyu': [(df_input['y']+(2/3)*df_input['x']) <= 3050000, (df['y']+(2/3)*df['x']) <= 3050000],
    }

    for key in tqdm(['hokkaidou', 'higashi', 'nishi', 'kyusyu']):
        df_input_condition = dict_condition[key][0]
        df_condition = dict_condition[key][1]

        df_input.loc[df_input_condition,
                     'h_z*'] = df_input.loc[df_input_condition, 'h_z']*(187.7875737003602)
        df.loc[df_condition, 'h_z*'] = df.loc[df_condition, 'h_z'] * \
            (187.7875737003602)
        df_input_xyz = df_input[df_input_condition][['x', 'y', 'h_z*']]
        df_xyz = df[df_condition][['x', 'y', 'h_z*']]

        pair_dist_xyz = cdist(df_xyz.values, df_input_xyz.values)
        t_list = []
        d_list = []
        pair_dist_xyz = np.where(pair_dist_xyz == 0, np.inf, pair_dist_xyz)
        t_values = df_input[df_input_condition]['t'].values
        for i in range(pair_dist_xyz.shape[0]):
            idx_min = np.argmin(pair_dist_xyz[i, :])
            t_list.append(t_values[idx_min])
            d_list.append(pair_dist_xyz[i, idx_min])

        df.loc[df_condition, 'near_t'] = t_list
        df.loc[df_condition, 'near_dist'] = d_list

    del df['h_z*']
    return df


def extra_split(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.1, shuffle=False)
    return train_data, test_data


def convert_output(output_data, debug=False):

    ret = output_data

    global log_std, log_mean

    ret = ret*log_std+log_mean

    ret = torch.exp(ret)

    if debug:
        print(ret)

    return ret

# 入力値の変換


def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)

    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)

def torch_select_column_by_condition(data, index, condition):
    condition_data = data[:, index]
    mask = condition_data.eq(condition)

    if len(torch.nonzero(mask)) == 0:
        return torch.Tensor()
    indices = torch.squeeze(torch.nonzero(mask), 1)
    select = torch.index_select(data, 0, indices)
    return select

def torch_select_column_by_condition_2(data, index, condition):
    condition_data = data[:, index]
    mask = condition_data.eq(condition)
    mask, mask_ = mask.min(1)
    if len(torch.nonzero(mask)) == 0:
        return torch.Tensor()
    indices = torch.squeeze(torch.nonzero(mask), 1)
    select = torch.index_select(data, 0, indices)
    return select

def create_d_list(est_data, z_unique, max_d):
    d_list = []
    bool_list = []
    for i in z_unique:
        pc = torch_select_column_by_condition(est_data, 2, i).double()
        pc = torch.sqrt(pairwise_distances(pc[:, [0, 1]]))
        bool_list_sub = []
        for j in range(pc.shape[0]):
            mask1 = pc[j, :] <= max_d
            mask2 = pc[j, :] > 0
            d_list.append(pc[j, :][mask1*mask2])
            bool_list_sub.append(mask1*mask2)
        bool_list.append(bool_list_sub)
    d_list = torch.cat(d_list)
    return d_list, bool_list


def create_t_list(pred_est_data, z_unique, bool_list):
    t_list = []
    for i, value in enumerate(z_unique):
        pc = torch_select_column_by_condition(pred_est_data, 2, value)
        for j in range(pc.shape[0]):
            vario_t = ((pc[j, 3]-pc[:, 3][[bool_list[i][j]]])**2)/2
            t_list.append(vario_t)
    t_list = torch.cat(t_list)
    return t_list


def region_condition(df):
    dict_condition = {
        'hokkaidou': [(df['x']-df['y']) <= 1200000, 40001],
        'higashi': [((df['x']-df['y']) > 1200000) & ((df['x']+df['y']) >= 5000000), 80001],
        'nishi': [((df['y']+(2/3)*df['x']) > 3050000) & ((df['x']+df['y']) < 5000000)],
        'kyusyu': [(df['y']+(2/3)*df['x']) <= 3050000, 80001],
    }
    return dict_condition

def region_condition_torch(ts):
    dict_condition = {
        'hokkaidou': [(ts[:, 0]-ts[:, 1]) <= 1200000, 40001],
        'higashi': [((ts[:, 0]-ts[:, 1]) > 1200000) & ((ts[:, 0]+ts[:, 1]) >= 5000000), 80001],
        'nishi': [((ts[:, 1]+(2/3)*ts[:, 0]) > 3050000) & ((ts[:, 0]+ts[:, 1]) < 5000000)],
        'kyusyu': [(ts[:, 1]+(2/3)*ts[:, 0]) <= 3050000, 80001],
    }
    return dict_condition


def calc_model_semivariogram_xy(df_input, region):
    features = ['x', 'y', 'h_z', 't']
    df_input = df_input[features]
    dict_conditon = region_condition(df_input)
    df_input = df_input[dict_conditon[region][0]]
    df_input['h_z'] = df_input['h_z'].round(-2)
    df_input = df_input.groupby(['x', 'y', 'h_z'], as_index=False).mean()
    z_unique = df_input['h_z'].unique()
    sv_all = []
    for z in z_unique:
        pc = df_input[df_input['h_z'] == z]
        pc_xy_dis = squareform(pdist(pc[['x', 'y']]))
        pc_t_vario = squareform((pdist(pc['t'].values.reshape(-1, 1))**2)/2)
        sep = 11000
        max_d = dict_conditon[region][1]
        sv_i = np.zeros(len(range(0, max_d, sep)))
        for i, value in enumerate(range(0, max_d, sep)):
            mask1 = pc_xy_dis > value
            mask2 = pc_xy_dis < value+sep
            mask = mask1*mask2
            res1 = pc_t_vario[mask]
            mask3 = res1 > 0
            res2 = res1[mask3]
            if res2.size:
                sv_i[i] = res2.mean()
            else:
                sv_i[i] = np.nan
        sv_all.append(sv_i)
    sv_all = np.vstack(sv_all)
    result = []
    for i in range(len(range(0, max_d, sep))):
        result.append(sv_all[:, i][~np.isnan(sv_all[:, i])].mean())
    return result

def create_d_bool_dict(ts):
    dict_conditon = region_condition_torch(ts)
    d_bool_dict = dict()
    for region in ['hokkaidou', 'higashi', 'kyusyu']:
        pc = ts[dict_conditon[region][0]]
        z = torch.unique(pc[:, 2])
        max_d = dict_conditon[region][1]
        d_list, bool_list = create_d_list(pc, z, max_d)
        d_bool_dict[region] = [d_list, bool_list]
    return d_bool_dict

def calc_pred_semivariogram_xy(ts, d_bool_dict, z_unique, region):
    dict_conditon = region_condition_torch(ts)
    ts = ts[dict_conditon[region][0]]
    sep = 11000
    max_d = dict_conditon[region][1]
    d_list, bool_list = d_bool_dict[region]
    t_list = create_t_list(ts, z_unique, bool_list)
    sv_list = []
    for i, value in enumerate(range(0, max_d, sep)):
        mask1 = d_list > value
        mask2 = d_list < value+sep
        mask = mask1*mask2
        res1 = t_list[mask]
        mask3 = res1 > 0
        res2 = (res1[mask3].mean())
        sv_list.append(res2)
    sv = torch.stack(sv_list)
    return sv

def convert_data(input_data, debug=False):
    ret = convert_output(model(input_data), debug)
    return ret


def Min_t(output, omin):
    min_t = ((output - omin).clamp(max=0) ** 2).mean()
    return min_t

def Max_t(output, omax):
    max_t = ((output - omax).clamp(min=0) ** 2).mean()
    return max_t

def Min_t_580(output):
    min_t = ((output - 580).clamp(max=0) ** 2).mean()
    return min_t

def Max_t_580(output):
    max_t = ((output - 580).clamp(min=0) ** 2).mean()
    return max_t

def Stacking_train_model(df_train_s,target_s):
    trn_s=df_train_s.drop(target_s,axis=1).reset_index(drop=True)
    tst_s=df_train_s[target_s].reset_index(drop=True)
    train_features_s=pd.DataFrame(np.zeros((trn_s.shape[0],3)))
    cv_s = KFold(n_splits=5, shuffle=True, random_state=0)
    for trn_idx_s, val_idx_s in cv_s.split(trn_s, tst_s):
        trn_x_s = trn_s.iloc[trn_idx_s, :]
        trn_y_s = tst_s[trn_idx_s]
        val_x_s = trn_s.iloc[val_idx_s, :]
        val_y_s = tst_s[val_idx_s]

        rf=RandomForestRegressor(random_state=0)
        lgbm=LGBMRegressor(random_state=0)
        xgb=XGBRegressor(random_state=0)

        rf.fit(trn_x_s, trn_y_s)
        lgbm.fit(trn_x_s, trn_y_s)
        xgb.fit(trn_x_s, trn_y_s)

        train_features_s.iloc[val_idx_s, 0]=rf.predict(val_x_s)
        train_features_s.iloc[val_idx_s, 1]=lgbm.predict(val_x_s)
        train_features_s.iloc[val_idx_s, 2]=xgb.predict(val_x_s)

    lr=LinearRegression()
    lr.fit(train_features_s,tst_s)
    return lr

def Stacking_est(model_s,df_train_s,target_s,df_est_s):
    trn_s=df_train_s.drop(target_s,axis=1)
    tst_s=df_train_s[target_s]

    rf=RandomForestRegressor(random_state=0)
    lgbm=LGBMRegressor(random_state=0)
    xgb=XGBRegressor(random_state=0)

    rf.fit(trn_s, tst_s)
    lgbm.fit(trn_s, tst_s)
    xgb.fit(trn_s, tst_s)

    est_features_s=pd.DataFrame(np.zeros((df_est_s.shape[0],3)))
    est_features_s[0]=rf.predict(df_est_s)
    est_features_s[1]=lgbm.predict(df_est_s)
    est_features_s[2]=xgb.predict(df_est_s)

    pred_s=model_s.predict(est_features_s)
    return pred_s

class Net(nn.Module):
    def __init__(self, input_size, output_size, unit_size):
        super(Net, self).__init__()
        self.fc_s = nn.Linear(input_size, unit_size)
        self.fc2 = nn.Linear(int(unit_size), int(unit_size))
        self.fc3 = nn.Linear(int(unit_size), int(unit_size))
        self.fc_e = nn.Linear(int(unit_size), output_size)

    def forward(self, x):

        x = F.relu(self.fc_s(x))
        # x=F.dropout(x,p=0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc3(x))
        # x=F.dropout(x,p=0.2)
        x = self.fc_e(x)

        return x

#%%
input_data = pd.read_csv('./input/useful_data/input_data.csv')
input_data = preprocess_input(input_data)
est_data = pd.read_csv('./input/useful_data/est_grid_500.csv')
est_data = preprocess_grid(est_data)
train_data=input_data.copy()
model_sv_dict = {}
for region in ['hokkaidou', 'higashi', 'kyusyu']:
    model_sv = np.array(
        calc_model_semivariogram_xy(train_data, region))
    model_sv_dict[region] = Variable(
        torch.from_numpy(model_sv).float()).to(device)
est_data_origin = est_data[['x', 'y', 'h_z']].values
est_data_origin = Variable(torch.from_numpy(
    est_data_origin).float()).to(device)
z_unique = torch.unique(est_data_origin[:, 2])
d_bool_dict = create_d_bool_dict(est_data_origin)
# %%
pred_est=pd.read_csv("./output/voxler/est_nk_inter100_0.csv")[["t"]].values
pred_est=Variable(torch.from_numpy(pred_est).float()).to(device)
pc_est = torch.cat((est_data_origin, pred_est), 1)
loss_sv = []
for region in ['hokkaidou', 'higashi', 'kyusyu']:
    sv = calc_pred_semivariogram_xy(
        pc_est, d_bool_dict, z_unique, region)
    loss_sv.append(torch.sqrt(
        F.mse_loss(sv, model_sv_dict[region])))
loss_sv = torch.mean(torch.stack(loss_sv))
print(loss_sv.data.item())
# %%
a=pd.read_csv("./output/voxler/est_nk_inter100_detail.csv")
a.t.max()
# %%
a[a.t>=374].sort_values("h_z",ascending=False)[(a.x==3450000) & (a.y==2000000)]
# %%
b=pd.read_csv("./input/useful_data/est_grid_detail.csv")
b[(b.x==3450000) & (b.y==2000000)]
# %%
c=c=pd.read_csv("./input/WGS_to_albers/grid_WGS_albers_detail.csv")
c
# %%
d=a.merge(c,on=["x","y"])
d
# %%
e=d[d.t>=374].sort_values("h_z",ascending=False).groupby(["x","y"],as_index=False).max()
e
# %%
import folium
import pandas as pd
# %%
def visualize_locations(df,  zoom=4):
    """日本を拡大した地図に、pandasデータフレームのlatitudeおよびlongitudeカラムをプロットする。
    """
        	
    # 図の大きさを指定する。
    f = folium.Figure(width=1000, height=500)

    # 初期表示の中心の座標を指定して地図を作成する。
    center_lat=34.686567
    center_lon=135.52000
    m = folium.Map([center_lat,center_lon], zoom_start=zoom).add_to(f)
        
    # データフレームの全ての行のマーカーを作成する。
    for i in range(0,len(df)):
        folium.CircleMarker(location=[df["ido"][i],df["keido"][i]],radius=1).add_to(m)
        
    return m

# %%
visualize_locations(e)
# %%
t_list=[]
for i in range(5):
    df=pd.read_csv(f'./output/voxler/est_dnn_inter_all_seed{i}.csv')
    t=df['t'].values
    t_list.append(t)
df['t']=np.vstack(t_list).std(axis=0)
df=df.rename(columns={"t":"std"})
df
df.to_csv('./output/voxler/est_dnn_inter_std.csv',index=False)
# %%
df["std"].max()
# %%
df[df.h_z==-1000]["std"].mean()
# %%
if False:
    print(1)
else:
    print(2)
# %%
import pandas as pd
a=pd.read_csv("./output/voxler/est_nk_inter100_detail.csv")
# %%
a[(a.x==3450000)&(a.y==2000000)]
# %%
import pandas as pd
input_data = pd.read_csv('./input/useful_data/input_data.csv')
est_data = pd.read_csv('./input/useful_data/est_grid_500.csv')
# %%
xy_unique=input_data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
xy_unique
# %%
idx_list=[]
for x,y in xy_unique.values:
    idx_list.append(input_data[(input_data.x==x) & (input_data.y==y)].index.tolist())
# %%
cv = KFold(n_splits=3, shuffle=True, random_state=0)
for trn_idx,val_idx in cv.split(xy_unique):
    l=[]
    for idx in trn_idx:
        l+=idx_list[idx]
    print(l)
# %%
a=input_data.copy()
b=input_data.copy()
# %%
pd.concat([a,b],axis=0)
# %%
pd.DataFrame(np.zeros((4,3)),columns=[0,"dnn","xgb"])
# %%
np.zeros((5,10))[1,:]
# %%
stacking_columns = [0,"dnn","rf"]
stacking_columns.index("rf")
# %%
def calc_model_semivariogram_z(df_input):
    df_input = df_input.copy()
    df_input.h_z=df_input["h_z"].round(-5)
    return df_input
input_data=pd.read_csv("./input/useful_data/input_data.csv")
# calc_model_semivariogram_z(input_data)
input_data.h_z
# %%
input_data.h_z.unique()
# %%
def torch_select_column_by_condition_2(data, index, condition):
    condition_data = data[:, index]
    mask1 = condition_data[:,0].eq(condition[0])
    mask2 = condition_data[:,1].eq(condition[1])
    mask=mask1*mask2
    if len(torch.nonzero(mask)) == 0:
        return torch.Tensor()
    indices = torch.squeeze(torch.nonzero(mask), 1)
    select = torch.index_select(data, 0, indices)
    return select
# %%
est_data_origin=est_data[["x","y","h_z"]].values
est_xy=est_data[["x","y","h_z"]].groupby(["x","y"],as_index=False).mean()[["x","y"]].values
est_xy
#%%
X_est=Variable(torch.from_numpy(est_data_origin).float())
XY_unique=Variable(torch.from_numpy(est_xy).float())
value=XY_unique[0]
value
# %%
condition_data=X_est[:,[0,1]]
condition_data
# %%
mask1=condition_data[:,0].eq(value[0])
mask2=condition_data[:,1].eq(value[1])
mask=mask1*mask2
# %%
indices = torch.squeeze(torch.nonzero(mask), 1)
indices
#%%
select = torch.index_select(X_est, 0, indices)
select
# %%
import numpy as np
np.sqrt(770)

# %%
import pandas as pd
# %%
input_data=pd.read_csv("./input/useful_data/input_data.csv")[["x","y","h_z","t"]]
input_data.h_z=input_data.h_z.round(-2)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
albers_wgs=pd.read_csv("./input/WGS_to_albers/input_WGS_albers.csv")
albers_wgs.x,albers_wgs.y=albers_wgs.x.round(),albers_wgs.y.round()
albers_wgs=albers_wgs.groupby(["x","y"],as_index=False).mean()
input_data=input_data.merge(albers_wgs,on=["x","y"],how="left")
input_data=input_data["ido,keido,h_z,t".split(",")]
input_data=input_data.rename(columns={"ido":"latitude","keido":"longitude"})
input_data
# %%
import folium
#%%

def visualize_locations(df,  zoom=4):
    """日本を拡大した地図に、pandasデータフレームのlatitudeおよびlongitudeカラムをプロットする。
    """
        	
    # 図の大きさを指定する。
    f = folium.Figure(width=400, height=400)

    # 初期表示の中心の座標を指定して地図を作成する。
    center_lat=	38.258595
    center_lon=137.6850225
    m = folium.Map([center_lat,center_lon], zoom_start=zoom).add_to(f)
        
    # データフレームの全ての行のマーカーを作成する。
    for i in range(0,len(df)):
        folium.CircleMarker(location=[df["latitude"][i],df["longitude"][i]],radius=1).add_to(m)
        
    return m
#%%
for i in range(500,-5000,-500):
    df=input_data[input_data.h_z==i].reset_index(drop=True)
    fig=visualize_locations(df)
    fig.save(f"./output/folium/index{i}.html")
# %%
# %%
list(range(500,-5000,-500))
# %%
input_data=pd.read_csv("./input/useful_data/input_data.csv")[["x","y","h_z","t"]]
input_data.h_z=input_data.h_z.round(-2)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
input_data
# %%

# %%
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform,cdist
import pykrige.variogram_models as vm
from tqdm import tqdm
import matplotlib.pyplot as plt
# %%
input_data=pd.read_csv("./input/useful_data/input_data.csv")[["x","y","h_z","t"]]
input_data.h_z=input_data.h_z.round(-2)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
for h_z in [0,-500,-1000,-1500]:
    df=input_data[input_data.h_z==h_z].reset_index(drop=True)
    xy=df[["x","y"]].values
    t=df[["t"]].values
    dist_xy=squareform(pdist(xy))
    dist_t=squareform(pdist(t)**2)
    sep=1000
    max_dist=5001
    res_list=[]
    for value in range(0,max_dist,sep):
        res=dist_t[(dist_xy>value) & (dist_xy<=value+sep)]
        res=res[res>0]
        res_list.append(res.mean())
    print(h_z)
    plt.plot(range(0,max_dist,sep),res_list)
    plt.show()
# %%
est_data=pd.read_csv("./input/useful_data/est_grid_500.csv")
est_data
# %%
est_data_nk_xy=[]
df=est_data.copy()
df.x = df.x+-3000
est_data_nk_xy

#%%
input_data=pd.read_csv("./input/useful_data/input_data.csv")[["x","y","h_z","t"]]
est_data=pd.read_csv("./input/useful_data/est_grid_500.csv")
est_data["t"]=np.random.randint(100,size = len(est_data))
est_data
#%%
def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)

    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)

#%%
def calc_model_semivariogram_z(df_input,df_est):
    df_input = df_input.copy()
    df_est = df_est.copy()
    df_input.h_z = df_input.h_z.round(-2)
    df_input_xy=df_input.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    df_est_xy=df_est.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    sv_all=[]
    for x,y in df_input_xy.values:
        pc=df_input[(df_input.x==x)&(df_input.y==y)]
        h_z=pc[["h_z"]].values
        t=pc[["t"]].values
        dist_z=squareform(pdist(h_z))
        dist_t=squareform(pdist(t))
        sv_i=np.zeros(len(range(500,1501,500)))
        for i,dist in enumerate(range(500,1501,500)):
            mask=dist_z==dist
            res=dist_t[mask]
            if res.size:
                sv_i[i]=res.mean()
            else:
                sv_i[i]=np.nan
        sv_all.append(sv_i)
    sv_all=np.vstack(sv_all)
    dist_nd=cdist(df_est_xy,df_input_xy)
    model_sv_list=[]
    for i in range(dist_nd.shape[0]):
        mask=dist_nd[i,:]<=1000
        model_sv_list.append(np.nanmean(sv_all[mask,:],axis=0))
    model_sv=np.vstack(model_sv_list)
    model_sv = Variable(torch.from_numpy(model_sv).float())
    return model_sv

model_sv=calc_model_semivariogram_z(est_data,est_data)
model_sv
#%%
est_xy=est_data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
est_data_origin=est_data[["x","y","h_z","t"]].values
X_est=Variable(torch.from_numpy(est_data_origin).float())
XY_unique=Variable(torch.from_numpy(est_xy.values).float())
#%%
def create_d_z_list(ts_est,ts_est_xy):
    dist_z_list=[]
    for x,y in ts_est_xy:
        pc=ts_est[(ts_est[:,0]==x) & (ts_est[:,1]==y)]
        h_z=pc[:,[2]]
        dist_z=torch.sqrt(pairwise_distances(h_z))
        dist_z_list.append(dist_z)
    return dist_z_list
dist_z_list = create_d_z_list(X_est,XY_unique)

def calc_pred_semivariogram_z(ts_est,ts_est_xy,dist_z_list):
    sv_all=[]
    for i,(x,y) in enumerate(ts_est_xy):
        pc=ts_est[(ts_est[:,0]==x) & (ts_est[:,1]==y)]
        pred_t=pc[:,[3]]
        dist_z=dist_z_list[i]
        dist_t=torch.sqrt(pairwise_distances(pred_t))
        sv_i=[]
        for z in range(500,1501,500):
            sv_i.append(dist_t[dist_z==z].mean())
        sv_i=torch.stack(sv_i)
        sv_all.append(sv_i)
    sv_all=torch.stack(sv_all)
    return sv_all
sv_all=calc_pred_semivariogram_z(X_est,XY_unique,dist_z_list)
sv_all
#%%
sv_loss=(model_sv-sv_all)**2
sv_loss=sv_loss[~sv_loss.isnan()].mean()
sv_loss
# %%
sep=20000
max_dist=70001
list(range(sep//2,max_dist,sep*2))
# %%
sep=20000
max_dist=70001
sv_i = []
for value in range(sep//2,max_dist,sep):
    print(value)

# %%
for i in range(10):
    value = i%4
    print(value)
# %%
for i in range(100):
    print(-random.randint(0,5)*1000)
# %%
est_data = pd.read_csv('./input/useful_data/est_grid_500.csv').sort_values(["x","y","h_z"],ascending=False)
est_data
# %%
np.array([1,2,3,4,5])[:2]
# %%
#%%
import pandas as pd
import numpy as np
# %%
def grid_albers(sep_xy,sep_z):#sep_xy(km),sep_z(m)#単位に注意
    #sep_xy(km),sep_z(m)#単位に注意
    
    #df1
    ido=range(950,1151,sep_xy)
    keido=range(2950,3051,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df1=pd.DataFrame(data,columns=['y','x','z'])
    #df2
    ido=range(800,1151,sep_xy)
    keido=range(3050,3251,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df2=pd.DataFrame(data,columns=['y','x','z'])
    #df3
    ido=range(1150,1451,sep_xy)
    keido=range(3100,3251,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df3=pd.DataFrame(data,columns=['y','x','z'])
    #df4
    ido=range(1050,1451,sep_xy)
    keido=range(3250,3351,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df4=pd.DataFrame(data,columns=['y','x','z'])
    #df5
    ido=range(1200,1751,sep_xy)
    keido=range(3350,3451,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df5=pd.DataFrame(data,columns=['y','x','z'])
    #df6
    ido=range(1200,2351,sep_xy)
    keido=range(3450,3551,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df6=pd.DataFrame(data,columns=['y','x','z'])
    #df7
    ido=range(1400,2201,sep_xy)
    keido=range(3550,3651,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df7=pd.DataFrame(data,columns=['y','x','z'])
    #df8
    ido=range(1400,2001,sep_xy)
    keido=range(3650,3751,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df8=pd.DataFrame(data,columns=['y','x','z'])
    #df9
    ido=range(2250,2751,sep_xy)
    keido=range(3300,3451,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df9=pd.DataFrame(data,columns=['y','x','z'])
    #df10
    ido=range(2400,2751,sep_xy)
    keido=range(3450,3651,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df10=pd.DataFrame(data,columns=['y','x','z'])
    #まとめ
    df=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10])
    df=df.loc[:,['x','y','z']]
    df['x']=df['x']-100000
    df['y']=df['y']-100000
    df=df.drop_duplicates(['x','y','z'])
    
    return df
# %%
df=grid_albers(2,1000)
df
# %%
df.groupby(["x","y"],as_index = False).mean()
# %%
est_data = pd.read_csv("./input/useful_data/est_grid_500.csv")
est_data
# %%
est_dataest_data.groupby(["x","y"],as_index=False).mean()
# %%
est_data
# %%
input_data = pd.read_csv("./input/useful_data/input_data.csv")
# %%
input_data
# %%
input_data[(input_data.h_z<=0) & (input_data.h_z>-1000)]
# %%
input_data[(input_data.h_z<=-1000) & (input_data.h_z>-2000)]
# %%
input_data[(input_data.h_z<=-2000) & (input_data.h_z>-3000)]
# %%
a=input_data.groupby(["x","y"],as_index=False).min()
a
# %%
input_data["label"]=pd.qcut(input_data.h_z,3,labels=["deep","mid","shallow"])
# %%
mid = input_data[input_data.label=="mid"]
mid
# %%
input_data=pd.read_csv("./input/useful_data/input_data.csv")
est_data=pd.read_csv("./input/useful_data/est_grid_500.csv")
est_data["t"]=np.random.randn(len(est_data))
input_data.h_z=input_data["h_z"].round(-2)
input_data["label"]=pd.qcut(input_data.h_z,3,labels=["deep","mid","shallow"])
input_data = input_data[input_data.label=="mid"]
input_xy=input_data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
est_xy=est_data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
input_sv_all=[]
for x,y in input_xy.values:
    pc=input_data[(input_data.x==x)&(input_data.y==y)]
    h_z=pc[["h_z"]].values
    t=pc[["t"]].values
    dist_z=squareform(pdist(h_z))
    dist_t=squareform(pdist(t))**2
    sv_i=np.zeros(len(range(0,1501,500)[1:]))
    for i,dist in enumerate(range(0,1501,500)[1:]):
        mask=dist_z==dist
        res=dist_t[mask]
        if res.size:
            sv_i[i]=res.mean()
        else:
            sv_i[i]=np.nan
    input_sv_all.append(sv_i)
input_sv_all=np.vstack(input_sv_all)
input_sv_all
# %%
input_sv_all[:,0][~np.isnan(input_sv_all[:,0])]
# %%
est_data=pd.read_csv("./input/useful_data/est_grid_500.csv")
est_data["t"]=np.random.randn(len(est_data))
est_xy=est_data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
XY_unique=Variable(torch.from_numpy(est_xy.values).double())
# %%
XY_unique

#%%
print(1)
# %%
def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)

    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)
# %%
pairwise_distances(XY_unique).detach().cpu().numpy()
# %%
cdist(est_xy,est_xy)**2
# %%
print(1)
# %%
def preprocess_input(df):
    df['t'] = np.where(df['t'].values <= 0, 0.1, df['t'].values)
    add_input_volcano = pd.read_csv('./input/volcano/add_input_volcano.csv')
    add_input_curie = pd.read_csv('./input/curie_point/add_input_curie.csv')
    add_input_tishitsu = pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
    add_input_onsen = pd.read_csv('./input/onsen/add_input_onsen.csv')
    add_input_kmeans = pd.read_csv('./input/k_means/add_input_kmeans.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])
    df = df.merge(add_input_kmeans, how='left', on=['x', 'y'])

    # df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df=pd.concat([df,tmp],axis=1)
    
    df=df.drop(['symbol','symbol_freq','formationAge_ja', 'group_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    df['age']=(df['age_a']+df['age_b'])/2

    return df

#%%
df = pd.read_csv('./input/useful_data/input_data.csv')
df = preprocess_input(df)
except_list = ["curie"]
df[["x","y"]+except_list+["volcano"]]
#%%
def permutation_importance(df,except_list):
    df = df.copy()
    df_plane = df.groupby(['x', 'y'], as_index=False).mean()
    df_plane.loc[:,except_list] = df_plane[except_list].sample(frac=1,random_state=0).reset_index(drop=True)
    df.drop(except_list,axis=1,inplace=True)
    df = df.merge(df_plane[["x","y"]+except_list],how="left",on=["x","y"])
    return df
# %%
permutation_importance(df[["x","y"]+except_list+["volcano"]],except_list).dropna()
#%%
input_data.loc[:,["x","y"]] = input_data[["x","y"]].sample(frac=1,random_state=0).reset_index(drop=True)
input_data
# %%
input_data = pd.read_csv('./input/useful_data/input_data.csv')
input_data

# %%
all_features_name_list=[]
features_name_list = ["volcano","curie","onsen","tishitsu","depth","grad"]
N=len(features_name_list)
for i in range(2**N):
    A=[]
    for j in range(N):
        if ((i>>j)&1):
            A.append(features_name_list[j])
    all_features_name_list.append(A)
all_features_name_list
# %%
features_dict = {
                "volcano":['volcano'],
                "curie":['curie'],
                "onsen":['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
                "tishitsu":['age_a', 'age_b','age','group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩'],
                "depth":[],#extraは800
                "grad":[],
                }
#%%
for all_f in all_features_name_list:
    features = ["x","y","h","z","h_z"]
    name = "basic"
    for f in all_f:
        features+=features_dict[f]
        name += "_"+f  
    print(features)
    print(name)
# %%
for i in ['volcano', 'curie', 'onsen']:
    print(i)
# %%
from sklearn.model_selection import train_test_split

#%%
def extra_split(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
    return train_data, test_data
#%%
input_data = pd.read_csv('./input/useful_data/input_data.csv')

train_data, test_data = extra_split(input_data)
# %%
train_data.h_z.min()
# %%
pd.read_csv('./input_japan/WGS_albers_h_ja_detail_.csv').dropna().to_csv('./input_japan/WGS_albers_h_ja_detail.csv',index = False)
# %%
a = pd.read_csv('./input_japan/grid_tishitsu_ja_detail_idokeido.csv')
a.isnull().sum()
# %%
212*100/a.shape[0]
# %%
for i in range(1):
    print(i)
# %%

#%%
fix_seed(0)
cv = KFold(n_splits=5, shuffle=True, random_state=0)
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
xy_data = input_data.groupby(
        ['x', 'y'], as_index=False).mean().loc[:, ['x', 'y']]
for num, (idx_trn, idx_tst) in enumerate(cv.split(xy_data)):
    # train test split
    trn_xy = xy_data.iloc[idx_trn, :].values
    tst_xy = xy_data.iloc[idx_tst, :].values
    train_data = pd.DataFrame()
    for x, y in trn_xy:
        trn_data = input_data[(input_data['x'] == x)
                                & (input_data['y'] == y)]
        train_data = pd.concat([train_data, trn_data], axis=0)
    test_data = pd.DataFrame()
    for x, y in tst_xy:
        tst_data = input_data[(input_data['x'] == x)
                                & (input_data['y'] == y)]
        test_data = pd.concat([test_data, tst_data], axis=0)
    print(test_data.shape[0]/input_data.shape[0])
# %%
lc_list=[]
hyper_list =[0.1915366765502127,
0.19733788835890165,
0.20986420867173736,
0.19222422757939064,
0.20903699883975763]
for i in range(5):
    lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_basic_{i}.csv')['test_loss'].values
    lc_list.append(hyper_list[i]*lc)
np.vstack(lc_list).sum(axis=0)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# %%
1.5*10
#%%
# 机①1000*1800
# 机②1000*1500
# 机②760*1530
#%%
1.5*1.53
#%%
# 暗室 1520(入口側)*2000
# プローバ箱型　1500+400*1000
# プローバ大 1800*1100
# プローバ小　900*1050
#%%
1.5*1.05
#%%
# 測定器ラック3つ　900*600
# 恒温槽 550*900
#%%
#%%
1.5*3
#%%
# 入口右ラック　1400*580

# 入口左ラック　1400*430

# 奥ラック 1200*610

# 左上デッドゾーン　1500*700

# 粟野先生ゾーン 2250*3000
#%%
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

# %%
pd.read_csv("./input_japan/useful_data/est_grid_detail_ja.csv").groupby(["x","y"],as_index=False).mean().to_csv("./Arcgis/grid.csv")
# %%
def preprocess_input(df):
    df['t'] = np.where(df['t'].values <= 0, 0.1, df['t'].values)
    add_input_volcano = pd.read_csv('./input_japan/volcano/add_input_volcano_ja.csv')
    add_input_curie = pd.read_csv('./input_japan/curie_point/add_input_curie_ja.csv')
    add_input_tishitsu = pd.read_csv('./input_japan/tishitsu/add_input_tishitsu_pred_ja.csv')
    add_input_onsen = pd.read_csv('./input_japan/onsen/add_input_onsen_ja.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])

    df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df=pd.concat([df,tmp],axis=1)
    
    df=df.drop(['symbol','symbol_freq','formationAge_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    df['age']=(df['age_a']+df['age_b'])/2

    return df
#%%
def extra_split(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.1, shuffle=False)
    return train_data, test_data
#%%
input_data = pd.read_csv("./input_japan/useful_data/input_data_ja.csv")#.groupby(["x","y"],as_index=False).max().mean()
input_data=preprocess_input(input_data)
train_data,test_data = extra_split(input_data)
a=train_data.groupby("group_ja").mean()[["t"]].rank(ascending=False).sort_values("t")
print(a)
# cv = KFold(n_splits=5, shuffle=True, random_state=0)
# xy_data = input_data.groupby(
#     ['x', 'y'], as_index=False).mean().loc[:, ['x', 'y']]
# for num, (idx_trn, idx_tst) in enumerate(cv.split(xy_data)):
#     trn_xy = xy_data.iloc[idx_trn, :].values
#     tst_xy = xy_data.iloc[idx_tst, :].values
#     train_data = pd.DataFrame()
#     for x, y in trn_xy:
#         trn_data = input_data[(input_data['x'] == x)
#                                 & (input_data['y'] == y)]
#         train_data = pd.concat([train_data, trn_data], axis=0)
#     test_data = pd.DataFrame()
#     for x, y in tst_xy:
#         tst_data = input_data[(input_data['x'] == x)
#                                 & (input_data['y'] == y)]
#         test_data = pd.concat([test_data, tst_data], axis=0)

#     a=train_data.groupby("group_ja").mean()[["t"]].rank(ascending=False).sort_values("t")
#     print(a)
# %%
input_data = pd.read_csv("./input_japan/useful_data/input_data_ja.csv")#.groupby(["x","y"],as_index=False).max().mean()
# input_data = preprocess_input(input_data)
input_data
#%%



#%%
a=input_data.groupby(["x","y"],as_index=False).min()
a[a.h_z<=-5000].shape
#%%
a=input_data.groupby("group_ja").mean()[["t"]].rank(ascending=False).sort_values("t")
# tishitsu_rank_dict
print(a)
#%%
input_data["group_rank"]=input_data["group_ja"].replace(tishitsu_rank_dict)
# %%
input_data = pd.read_csv("./input_japan/useful_data/input_data_ja.csv")#.groupby(["x","y"],as_index=False).max().mean()
input_data.h_z=input_data.h_z.round(-1)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
input_data[input_data.h_z==500].shape
# %%
def extra_split(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
    return train_data, test_data
# %%
df,_=extra_split(input_data)
df
# %%
df.min()
# %%
a=np.ones((10,3))
np.mean([a,2*a,3*a],axis=0)
# %%
b=""
print(f"a{b}c")
# %%

def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('./input_japan/volcano/add_grid_volcano_detail_ja.csv')
    add_grid_curie = pd.read_csv('./input_japan/curie_point/add_grid_curie_detail_ja.csv')
    add_grid_tishitsu = pd.read_csv('./input_japan/tishitsu/add_grid_tishitsu_detail_pred_ja.csv')
    add_grid_onsen = pd.read_csv('./input_japan/onsen/add_grid_onsen_detail_ja.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])

    # df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df=pd.concat([df,tmp],axis=1)
    
    df=df.drop(['symbol','symbol_freq','formationAge_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)#, 'group_ja'
    df['age']=(df['age_a']+df['age_b'])/2

    return df
# %%
est_data = pd.read_csv('./input_japan/useful_data/est_grid_500_ja.csv').sort_values(["x","y","h_z"])
est_data=est_data.groupby(["x","y"],as_index=False).mean()
est_data  =preprocess_grid(est_data)
# est_data[["x","y","h_z","age"]].to_csv("./input_japan/tishitsu/database/tishitsu_age.csv",index=False)
# %%
est_data[['group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩']].sum()
# %%
a=""
print(f"exe{a}100")
# %%
import codecs
with codecs.open(rf"./input_japan/volcano/database/volcano_hyoukou.csv",'r','shift-jis','ignore') as f:
    df=pd.read_csv(f,names=["a","b","c","d","e"])
df
# %%
df=df[df.e>0]
df.sort_values("e")
# %%
# %%
from numpy.lib.ufunclike import fix
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from optuna.samplers import TPESampler
import optuna
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from xgboost.callback import LearningRateScheduler
import time
import datetime
print(datetime.datetime.now())
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
#%%
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(0)
# %%

def preprocess_input(df):
    df['t'] = np.where(df['t'].values <= 0, 0.1, df['t'].values)
    add_input_volcano = pd.read_csv('./input_japan/volcano/add_input_volcano_ja.csv')
    add_input_curie = pd.read_csv('./input_japan/curie_point/add_input_curie_ja.csv')
    add_input_tishitsu = pd.read_csv('./input_japan/tishitsu/add_input_tishitsu_pred_ja.csv')
    add_input_onsen = pd.read_csv('./input_japan/onsen/add_input_onsen_ja.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])

    # df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df=pd.concat([df,tmp],axis=1)
    
    df=df.drop(['symbol','symbol_freq','formationAge_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)#, 'group_ja'
    df['age']=(df['age_a']+df['age_b'])/2

    return df
def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('./input_japan/volcano/add_grid_volcano_detail_ja.csv')
    add_grid_curie = pd.read_csv('./input_japan/curie_point/add_grid_curie_detail_ja.csv')
    add_grid_tishitsu = pd.read_csv('./input_japan/tishitsu/add_grid_tishitsu_detail_pred_ja.csv')
    add_grid_onsen = pd.read_csv('./input_japan/onsen/add_grid_onsen_detail_ja.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])

    # df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df=pd.concat([df,tmp],axis=1)
    
    df=df.drop(['symbol','symbol_freq','formationAge_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)#, 'group_ja'
    df['age']=(df['age_a']+df['age_b'])/2

    return df
# %%
features_dict = {
                "volcano":['volcano'],
                "curie":['curie'],
                "onsen":['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
                "tishitsu_ohe":['age_a', 'age_b','age'],
                "tishitsu_rank":['age_a', 'age_b','age',"group_rank"],
                "depth":["depth0","depth500","depth1000"],#extraは800
                "grad":['grad','grad_max','grad_min','grad_max_h_z','grad_min_h_z'],
                }
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)
master_features=['x', 'y', 'h', 'z', 'h_z']
    # all_f = ["curie","onsen","tishitsu_rank","depth","grad"]
all_f = ["volcano","curie","onsen","tishitsu_ohe","depth","grad"]
name = "basic"
for f in all_f:
    master_features+=features_dict[f]
curie_data = pd.read_csv('./input_japan/curie_point/grid_curie_ja.csv')
curie_data = preprocess_grid(curie_data)
est_data_detail = pd.read_csv('./input_japan/useful_data/est_grid_detail_ja.csv').sort_values(["x","y","h_z"])
est_data_detail = preprocess_grid(est_data_detail)


add_input_depth = pd.read_csv(f'./input_japan/depth/add_input_depth_ja.csv')
add_grid_depth = pd.read_csv(f'./input_japan/depth/add_grid_depth_detail_ja.csv')

input_data=input_data.merge(add_input_depth, how='left', on=['x', 'y'])
curie_data=curie_data.merge(add_grid_depth, how='left', on=['x', 'y'])
est_data_detail=est_data_detail.merge(add_grid_depth, how='left', on=['x', 'y'])

add_input_grad = pd.read_csv(f'./input_japan/grad/add_input_grad_ja.csv')
add_grid_grad = pd.read_csv(f'./input_japan/grad/add_grid_grad_detail_ja.csv')
input_data=input_data.merge(add_input_grad, how='left', on=['x', 'y'])
curie_data=curie_data.merge(add_grid_grad, how='left', on=['x', 'y'])
est_data_detail=est_data_detail.merge(add_grid_grad, how='left', on=['x', 'y'])
#%%
name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
est_data_t=pd.read_csv(f'./output_japan_last/voxler/nk/est_nk500_output_{name}_detail.csv')#.groupby(["x","y"],as_index=False).mean()[["x","y","t"]]
est_data_t
#%%
est_data_detail=est_data_detail[est_data_detail.z>=0].reset_index(drop=True)
a=est_data_detail[est_data_detail.h_z==0]
plt.scatter(a.x,a.y,c="k",s=0.1)
#%%
est_data=est_data_detail.copy()
# input_data = input_data.groupby(["x","y"],as_index=False).mean()
# est_data = est_data.groupby(["x","y"],as_index=False).mean()
# est_data_t=pd.read_csv(f'./output_japan_last/voxler/nk/est_nk500_output_{name}_detail.csv').groupby(["x","y"],as_index=False).mean()[["x","y","t"]]
# est_data=est_data.merge(est_data_t,how="left",on=["x","y"])

est_data=est_data.merge(est_data_t,how="left",on=["x","y","h_z"])

X_train = input_data[["volcano","curie",'age','group_ja_堆積岩', 'group_ja_火成岩',"Temp","Na","Cl","grad","grad_max","grad_max_h_z","depth0","depth500","depth1000"]].values
Y_train = input_data[["t"]].values

X_est = est_data[["volcano","curie",'age','group_ja_堆積岩', 'group_ja_火成岩',"Temp","Na","Cl","grad","grad_max","grad_max_h_z","depth0","depth500","depth1000"]].values
Y_est = est_data[["t"]].values

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_est = scaler.transform(X_est)
import pykrige.variogram_models as vm
X_train
#%%
def variogram(x_train,y_train, sep=11000, max_dist=500001, parameters={'sill': 9, 'range': 200000, 'nugget': 0.5}, how='gaussian'):
    xy_dis = squareform(pdist(x_train))*100
    t_vario = squareform(pdist(y_train.reshape(-1, 1))**2)
    sv_i = np.zeros(len(range(0, max_dist, sep)))
    sill_, range_, nugget_ = parameters['sill'], parameters['range'], parameters['nugget']
    for i, value in enumerate(tqdm(range(0, max_dist, sep))):
        mask1 = xy_dis > value
        mask2 = xy_dis < value+sep
        mask = mask1*mask2
        res1 = t_vario[mask]
        mask3 = res1 > 0
        res2 = (res1[mask3].mean())/2
        sv_i[i] = res2
    x = range(0, max_dist, sep)
    plt.plot(x[:], sv_i[:], marker='o')#c='black', 
    if how == 'gaussian':
        plt.plot(x[:], vm.gaussian_variogram_model(
            [sill_, range_, nugget_], np.array(x))[:], c='red')
    elif how == 'spherical':
        plt.plot(x[:], vm.spherical_variogram_model(
            [sill_, range_, nugget_], np.array(x))[:], c='red')
    # print(df.columns[2])
    # plt.show()
#%%
variogram(X_train,Y_train,25,150,parameters={'sill': 2500-250, 'range': 110, 'nugget': 250},how='ga')
variogram(X_est,Y_est,25,150,parameters={'sill': 2500-250, 'range': 110, 'nugget': 250},how='ga')
#%%
input_data.h_z=input_data.h_z.round(-1)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
h_z_list=np.sort(np.unique(input_data.h_z.values))
legend_list = []
for h_z in [-5000,-4000,-3000,-2000,-1000,-500,0]:
    df=input_data[input_data.h_z==h_z].reset_index(drop=True)
    df_est =est_data[est_data.h_z==h_z].reset_index(drop=True)
    # if df.shape[0]<400:
    #     continue
    X_train = df[["volcano","curie",'age','group_ja_堆積岩', 'group_ja_火成岩',"Temp","Na","Cl","grad","grad_max","grad_max_h_z","depth0","depth500","depth1000"]].values
    Y_train = df[["t"]].values
    X_est = df_est[["volcano","curie",'age','group_ja_堆積岩', 'group_ja_火成岩',"Temp","Na","Cl","grad","grad_max","grad_max_h_z","depth0","depth500","depth1000"]].values
    Y_est = df_est[["t"]].values
    scaler = MinMaxScaler()
    X_est = scaler.fit_transform(X_est)
    # print(h_z,df.shape)
    # if h_z in [-1000,-500,0]:
    #     scaler = MinMaxScaler()
    #     X_train = scaler.fit_transform(X_train)
    #     variogram(X_train,Y_train,25,150,parameters={'sill': 2500-250, 'range': 110, 'nugget': 250},how='ga')
    #     legend_list.append(f"{int(h_z)},{df.shape[0]}")
    variogram(X_est,Y_est,25,150,parameters={'sill': 2500-250, 'range': 110, 'nugget': 250},how='ga')
    legend_list.append(f"est{int(h_z)}")
plt.legend(legend_list,loc="lower right",bbox_to_anchor=(1.3, 0))
    
#%%
plt.hist(pdist(X_train))
#%%
plt.hist(pdist(X_est))
#%%
data = pd.concat([input_data[master_features+["t"]],curie_data[master_features+["t"]]])
data
# %%
X_train=data[master_features]
X_test=data["t"]
X_est = est_data_detail[master_features]
# %%
lgbm=LGBMRegressor(random_state=0)
lgbm.fit(X_train,X_test)
est_data_detail["t"]=lgbm.predict(X_est)


# %%
import seaborn as sns
sns.set()
fig,ax=plt.subplots()
# ax.set_xlabel()
# ax.set_ylabel()
xy=est_data_detail.groupby(["x","y"],as_index=False).mean()[["x","y"]]
for i,(x,y) in enumerate(xy.values):
    pc=est_data_detail[(est_data_detail.x==x) & (est_data_detail.y==y)]
    ax.plot(pc.t,pc.h_z,"black")
    if i==100:
        break
# %%
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)

pred_est_list = []
learning_curve_list_master=[]

xy_data = input_data.groupby(
    ['x', 'y'], as_index=False).mean().loc[:, ['x', 'y']]
for num, (idx_trn, idx_tst) in enumerate(cv.split(xy_data)):
    if num == args.number:
        fix_seed(0)
        print(num)
        master_features=['x', 'y', 'h', 'z', 'h_z']
        name = "basic"
        for f in all_f:
            master_features+=features_dict[f]
            name += "_"+f 
        print(name)
        cat_features = ['group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩']
        # if "tishitsu_ohe" in name:
        # preprocess
        est_data = pd.read_csv('./input_japan/useful_data/est_grid_500_ja.csv')
        curie_data = pd.read_csv('./input_japan/curie_point/grid_curie_ja.csv')
        curie_data_580ika = pd.read_csv('./input_japan/curie_point/grid_curie_580ika_ja.csv')
        curie_data_580izyou = pd.read_csv('./input_japan/curie_point/grid_curie_580izyou_ja.csv')

        est_data = preprocess_grid(est_data)
        curie_data = preprocess_grid(curie_data)
        curie_data_580ika = preprocess_grid(curie_data_580ika)
        curie_data_580izyou = preprocess_grid(curie_data_580izyou)

        # train test split
        trn_xy = xy_data.iloc[idx_trn, :].values
        tst_xy = xy_data.iloc[idx_tst, :].values
#%%
import requests

class LINENotifyBot(object):
    API_URL = 'https://notify-api.line.me/api/notify'
    def __init__(self, access_token):
        self.__headers = {'Authorization': 'Bearer ' + access_token}

    def send(
        self,
        message,
        image=None,
        sticker_package_id=None,
        sticker_id=None,
    ):
        payload = {
            'message': message,
            'stickerPackageId': sticker_package_id,
            'stickerId': sticker_id,
        }
        files = {}
        if image != None:
            files = {'imageFile': open(image, 'rb')}
        r = requests.post(
            LINENotifyBot.API_URL,
            headers=self.__headers,
            data=payload,
            files=files,
        )
bot = LINENotifyBot(access_token="Vur0KYsEIKUU7vSpAPsujaFFovHM9a0Olrudcn1xEgG")

bot.send(123)

# %%
