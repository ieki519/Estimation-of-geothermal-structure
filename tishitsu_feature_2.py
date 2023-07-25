#%%
# if 0:#grid
#     import pandas as pd
#     import re
#     import requests
#     import time
#     import json
#     xy_grid_wgs=pd.read_csv('./input/elevation/hyoukou_wgs.csv')[['ido','keido']]
#     all_tishitsu=[]
#     time_now=time.time()
#     for ido,keido in xy_grid_wgs.values:
#         url=f'https://gbank.gsj.jp/seamless/v2/api/1.2.1/legend.json?point={ido},{keido}'
#         res=requests.get(url)
#         tishitsu=res.text
#         dict_tishitsu=json.loads(tishitsu)
#         df_tishitsu=pd.DataFrame(dict_tishitsu.values(),index=dict_tishitsu.keys()).T
#         df_tishitsu['ido'],df_tishitsu['keido']=ido,keido
#         df_tishitsu=df_tishitsu[['ido','keido','symbol','formationAge_ja','group_ja','lithology_ja']]
#         all_tishitsu.append(df_tishitsu)
#         time.sleep(5)
#     all_tishitsu=pd.concat(all_tishitsu).reset_index(drop=True)
#     all_tishitsu.to_csv('./input/tishitsu/add_grid_tishitsu_idokeido.csv',index=False)

#     all_tishitsu.loc[:,['ido','keido']]=all_tishitsu[['ido','keido']].round(3)
#     grid_wgs_albers=pd.read_csv('./input/WGS_to_albers/grid_WGS_albers.csv')
#     grid_wgs_albers['ido'],grid_wgs_albers['keido']=grid_wgs_albers['ido'].round(3),grid_wgs_albers['keido'].round(3)
#     all_tishitsu=all_tishitsu.merge(grid_wgs_albers,how='left',on=['ido','keido'])
#     all_tishitsu=all_tishitsu[['x','y','symbol','formationAge_ja','group_ja','lithology_ja']]
#     all_tishitsu.to_csv('./input/tishitsu/add_grid_tishitsu_.csv',index=False)
#     print((time.time()-time_now))
#%%
# if 0:#input
#     import pandas as pd
#     import re
#     import requests
#     import time
#     import json
#     xy_input_wgs=pd.read_csv('./input/WGS_to_albers/input_WGS_albers.csv')[['ido','keido']]
#     all_tishitsu=[]
#     time_now=time.time()
#     for ido,keido in xy_input_wgs.values:
#         url=f'https://gbank.gsj.jp/seamless/v2/api/1.2.1/legend.json?point={ido},{keido}'
#         res=requests.get(url)
#         tishitsu=res.text
#         dict_tishitsu=json.loads(tishitsu)
#         df_tishitsu=pd.DataFrame(dict_tishitsu.values(),index=dict_tishitsu.keys()).T
#         df_tishitsu['ido'],df_tishitsu['keido']=ido,keido
#         df_tishitsu=df_tishitsu[['ido','keido','symbol','formationAge_ja','group_ja','lithology_ja']]
#         all_tishitsu.append(df_tishitsu)
#         time.sleep(5)
#     all_tishitsu=pd.concat(all_tishitsu).reset_index(drop=True)
#     # all_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu_idokeido.csv',index=False)
#     #%%
#     all_tishitsu=pd.read_csv('./input/tishitsu/add_input_tishitsu_idokeido.csv')
#     all_tishitsu.loc[:,['ido','keido']]=all_tishitsu[['ido','keido']].round(8)

#     input_wgs_albers=pd.read_csv('./input/WGS_to_albers/input_WGS_albers.csv')
#     input_wgs_albers['ido'],input_wgs_albers['keido']=input_wgs_albers['ido'].round(8),input_wgs_albers['keido'].round(8)
#     input_wgs_albers=input_wgs_albers[['ido','keido','x','y']]

#     all_tishitsu=all_tishitsu.merge(input_wgs_albers,on=['ido','keido'],how='left')
#     all_tishitsu=all_tishitsu[['x','y','symbol','formationAge_ja','group_ja','lithology_ja']]
#     # all_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu.csv',index=False)
#     print((time.time()-time_now))

# add_input_tishitsu=pd.read_csv('./input/tishitsu/add_input_tishitsu.csv')
# add_input_tishitsu['x'],add_input_tishitsu['y']=add_input_tishitsu['x'].round(),add_input_tishitsu['y'].round()
# add_input_tishitsu=add_input_tishitsu.drop_duplicates().reset_index(drop=True)
# add_input_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu.csv',index=False)
# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import folium
# %%
from numpy.core.fromnumeric import mean
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
import time
# import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from sklearn.linear_model import 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error
import pykrige.variogram_models as vm
from sklearn.ensemble import VotingClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# %%
from numpy.core.fromnumeric import mean
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
import time
# import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error
import pykrige.variogram_models as vm
from sklearn.ensemble import VotingRegressor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#%%
add_input_tishitsu=pd.read_csv('./input/tishitsu/database/add_input_tishitsu.csv')
add_grid_tishitsu_detail=pd.read_csv('./input/tishitsu/database/add_grid_tishitsu_detail_.csv')
# f = folium.Figure(width=1000, height=500)
# f = folium.Figure(width=1000, height=500)
# center_lat=34.686567
# center_lon=135.52000
# m = folium.Map([center_lat,center_lon], zoom_start=4).add_to(f)
# for ido,keido in add_grid_tishitsu[add_grid_tishitsu['symbol'].isnull()][['ido','keido']].values:
#     folium.Marker(location=[ido,keido]).add_to(m)
# m
add_grid_tishitsu_detail
#%%

def preprocess_input(df):
    df['t'] = np.where(df['t'].values <= 0, 0.1, df['t'].values)
    add_input_volcano = pd.read_csv('./input/volcano/add_input_volcano.csv')
    add_input_curie = pd.read_csv('./input/curie_point/add_input_curie.csv')
    add_input_tishitsu = pd.read_csv('./input/tishitsu/database/add_input_tishitsu.csv')
    add_input_onsen = pd.read_csv('./input/onsen/add_input_onsen.csv')
    # add_input_kmeans = pd.read_csv('./input/k_means/add_input_kmeans.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])
    # df = df.merge(add_input_kmeans, how='left', on=['x', 'y'])

    # df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    # tmp = pd.get_dummies(df["group_ja"], "group_ja")
    # df=pd.concat([df,tmp],axis=1)
    
    # df=df.drop(['symbol','symbol_freq','formationAge_ja', 'group_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    # df['age']=(df['age_a']+df['age_b'])/2

    return df

def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('./input/volcano/add_grid_volcano_detail.csv')
    add_grid_curie = pd.read_csv('./input/curie_point/add_grid_curie_detail.csv')
    add_grid_tishitsu = pd.read_csv('./input/tishitsu/database/add_grid_tishitsu_detail_.csv')
    add_grid_onsen = pd.read_csv('./input/onsen/add_grid_onsen_detail.csv')
    # add_grid_kmeans = pd.read_csv('./input/k_means/add_grid_kmeans_detail.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])
    # df = df.merge(add_grid_kmeans, how='left', on=['x', 'y'])

    # df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    # tmp = pd.get_dummies(df["group_ja"], "group_ja")
    # df=pd.concat([df,tmp],axis=1)
    
    # df=df.drop(['symbol','symbol_freq','formationAge_ja', 'group_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    # df['age']=(df['age_a']+df['age_b'])/2
    return df
#%%
input_data=pd.read_csv('./input/useful_data/input_data.csv').groupby(['x','y'],as_index=False).mean()
est_grid_detail=pd.read_csv('./input/useful_data/est_grid_detail.csv').groupby(['x','y'],as_index=False).mean()
add_input_tishitsu = preprocess_input(input_data)
add_grid_tishitsu_detail = preprocess_grid(est_grid_detail)
import codecs
with codecs.open(rf'./input/tishitsu/地質年代.csv','r','shift-jis','ignore') as f:
    age=pd.read_csv(f)
add_input_tishitsu=add_input_tishitsu.merge(age,on=['formationAge_ja'],how='left')
add_grid_tishitsu_detail=add_grid_tishitsu_detail.merge(age,on=['formationAge_ja'],how='left')
add_grid_tishitsu_detail
#%%
tishitsu=pd.concat([add_input_tishitsu,add_grid_tishitsu_detail])

train_tishitsu=tishitsu[~tishitsu['symbol'].isnull()]
test_input_tishitsu=add_input_tishitsu[add_input_tishitsu['symbol'].isnull()]
test_grid_tishitsu=add_grid_tishitsu_detail[add_grid_tishitsu_detail['symbol'].isnull()]
#%%
train_tishitsu.columns
#%%
tishitsu_features = ['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie','Temp', 'pH', 'Na', 'K',
       'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion']
#%%
for target in ['age_a',"age_b"]:
    print(target)
    trn = train_tishitsu[tishitsu_features]
    tst = train_tishitsu[target]
    MLA = {'xgb': XGBRegressor(), 'lgbm': LGBMRegressor(), 'lr': LinearRegression(
        ),  'dt': DecisionTreeRegressor(), 'rf': RandomForestRegressor(), 'kn': KNeighborsRegressor()}
    for m in MLA.keys():
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        result = model_selection.cross_validate(
            MLA[m], trn, tst, cv=cv, scoring='neg_mean_squared_error')
        print(m, np.sqrt(-result['test_score'].mean()))
#%%
for target in ['group_ja']:
    print(target)
    trn = train_tishitsu[tishitsu_features]
    train_tishitsu.loc[:,target],_ = train_tishitsu[target].factorize(na_sentinel=-99)
    tst = train_tishitsu[target].copy()
    num_class = tst.unique().shape[0]
    MLA = {'xgb': XGBClassifier(objective='multi:softprob',eval_metric='mlogloss',num_class=num_class), 'lgbm': LGBMClassifier(), 'dt': DecisionTreeClassifier(), 'rf': RandomForestClassifier(), 'kn': KNeighborsClassifier()}
    for m in MLA.keys():
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        result = model_selection.cross_validate(
            MLA[m], trn, tst, cv=cv, scoring="accuracy")
        print(m, result['test_score'].mean())
#%%
input_data.describe()
#%%
for target in ['symbol','formationAge_ja', 'group_ja', 'lithology_ja']:
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_tishitsu.loc[:,['x','y','h']],train_tishitsu[target])
    test_input_tishitsu.loc[:,target]=knn.predict(test_input_tishitsu.loc[:,['x','y','h']])
    test_grid_tishitsu.loc[:,target]=knn.predict(test_grid_tishitsu.loc[:,['x','y','h']])

add_input_tishitsu=pd.concat([add_input_tishitsu.dropna(),test_input_tishitsu]).reset_index(drop=True)
add_grid_tishitsu_detail=pd.concat([add_grid_tishitsu_detail.dropna(),test_grid_tishitsu]).reset_index(drop=True)

add_input_tishitsu=add_input_tishitsu[['x', 'y','symbol','formationAge_ja', 'group_ja', 'lithology_ja']]
add_grid_tishitsu_detail=add_grid_tishitsu_detail[['x', 'y','symbol','formationAge_ja', 'group_ja', 'lithology_ja']]

# add_input_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu_pred.csv',index=False)
# add_grid_tishitsu_detail.to_csv('./input/tishitsu/add_grid_tishitsu_detail_pred.csv',index=False)

# %%


add_input_tishitsu=pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
add_grid_tishitsu_detail=pd.read_csv('./input/tishitsu/add_grid_tishitsu_detail_pred.csv')
tishitsu=pd.concat([add_input_tishitsu,add_grid_tishitsu_detail])
# pd.DataFrame(np.sort(tishitsu['formationAge_ja'].unique())).to_csv('./成形前データ/地質年代.csv',index=False,encoding='cp932')
# %%
import codecs
with codecs.open(rf'./input/tishitsu/地質年代.csv','r','shift-jis','ignore') as f:
    age=pd.read_csv(f)
age
# %%
add_input_tishitsu=add_input_tishitsu.merge(age,on=['formationAge_ja'],how='left')
add_grid_tishitsu_detail=add_grid_tishitsu_detail.merge(age,on=['formationAge_ja'],how='left')
add_grid_tishitsu_detail
# %%
def encode_top_100(s):#Seriesを入力
    uniqs,freqs=np.unique(s,return_counts=True)
    top=sorted(zip(uniqs,freqs),key=lambda x:x[1],reverse=True)
    top_map={uf[0]:lank for uf,lank in zip(top,range(len(top)))}
    return s.map(lambda x:top_map.get(x,0)).astype(np.int),top_map
# %%
add_grid_tishitsu_detail['symbol_freq'],freq_dict=encode_top_100(add_grid_tishitsu_detail['symbol'])
add_input_tishitsu['symbol_freq']=add_input_tishitsu['symbol'].map(lambda x:freq_dict.get(x,0))
add_grid_tishitsu_detail['group_freq'],freq_dict=encode_top_100(add_grid_tishitsu_detail['group_ja'])
add_input_tishitsu['group_freq']=add_input_tishitsu['group_ja'].map(lambda x:freq_dict.get(x,0))
add_grid_tishitsu_detail['lithology_freq'],freq_dict=encode_top_100(add_grid_tishitsu_detail['lithology_ja'])
add_input_tishitsu['lithology_freq']=add_input_tishitsu['lithology_ja'].map(lambda x:freq_dict.get(x,0))
# add_input_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu_pred.csv',index=False)
# add_grid_tishitsu_detail.to_csv('./input/tishitsu/add_grid_tishitsu_detail_pred.csv',index=False)
# %%
# %%
import seaborn as sns
#%%
add_grid_tishitsu_detail['lithology_ja'].unique().shape
# %%
sns.countplot(x='lithology_ja',data=add_input_tishitsu)
# %%
