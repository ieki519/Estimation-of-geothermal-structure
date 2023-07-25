#%%
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform,cdist
import pykrige.variogram_models as vm
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import winsound
import matplotlib.cm as cm
import time
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(device)
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
def extra_split(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.1, shuffle=False)
    return train_data, test_data
#%%
est_data=pd.read_csv("./input/useful_data/est_grid_500.csv")
est_data
# %%
est_data_nk_xy=[]
for value in [-2000,2000,-6000,6000]:
    df=est_data.copy()
    df.x = df.x+value
    est_data_nk_xy.append(df)
for value in [-2000,2000,-6000,6000]:
    df=est_data.copy()
    df.y = df.y+value
    est_data_nk_xy.append(df)
est_data_nk_xy = pd.concat(est_data_nk_xy)
est_data_nk_xy
# %%
est_data_nk_xy.groupby(["x","y"],as_index=False).mean()[["x","y"]]
# %%
h=est_data_nk_xy.h.values
h=h.reshape(8,est_data.shape[0])
h - np.ones((1,est_data.shape[0]))
# %%
np.ones((1,est_data.shape[0]))
# %%
input_data=pd.read_csv("./input_japan/useful_data/input_data_ja.csv")[["x","y","h_z","t"]]
# geothermal_db=pd.read_csv('./input/useful_data/database/geothermal_db.csv')
# profile_db=pd.read_csv("./input/useful_data/database/profile_db.csv")
# input_data = pd.concat([profile_db,geothermal_db]).reset_index(drop=True)
# input_data = pd.read_csv("./input/useful_data/database/profile_db.csv")

input_data.h_z=input_data.h_z.round(-2)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
h_z_list=np.sort(np.unique(input_data.h_z.values))
# h_z_list=h_z_list[h_z_list>=-1600]
res_list_master=[]
for h_z in h_z_list:
    df=input_data[input_data.h_z==h_z].reset_index(drop=True)
    xy=df[["x","y"]].values
    t=df[["t"]].values
    dist_xy=squareform(pdist(xy))
    dist_t=squareform(pdist(t)**2)
    sep=1100
    max_dist=30001
    res_list=[]
    for value in range(sep//2,max_dist,sep):
        res=dist_t[(dist_xy>=value) & (dist_xy<value+sep)]
        res=res[res>0]
        res_list.append(res.mean())
    res_list_master.append(res_list)
    print(h_z)
    # plt.plot(range(sep//2,max_dist,sep),res_list,marker = ".")
    # plt.show()
print("-"*50)
plt.plot(range(sep//2,max_dist,sep),np.nanmean(np.array(res_list_master),axis=0))
plt.show()
# %%
est_data=pd.read_csv("./input/useful_data/est_grid_500.csv")
est_data_origin_xy=est_data.groupby(["x","y"],as_index=False).mean()[["x","y"]].values
est_data_origin_xy
# %%
input_data=pd.read_csv("./input/useful_data/input_data.csv")
input_data = preprocess_input(input_data)
input_data , _ = extra_split(input_data)
# geothermal_db=pd.read_csv('./input/useful_data/database/geothermal_db.csv')
# profile_db=pd.read_csv("./input/useful_data/database/profile_db.csv")
# input_data = pd.concat([profile_db,geothermal_db]).reset_index(drop=True)
# input_data = pd.read_csv("./input/useful_data/database/profile_db.csv")

# input_data.h_z = input_data.h_z.round(-2)
# input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
# input_data[input_data.h_z==0].sort_values(["x","y"]).reset_index(drop=True)
input_data
#%%
input_data_origin_xy=input_data.groupby(["x","y"],as_index=False).mean()[["x","y"]].values
input_data_origin_xy
# %%
def calc_model_semivariogram_xy(df_input,df_est):
    model_sv = []
    df_input = df_input.copy()
    df_est = df_est.copy()
    df_input.h_z = df_input.h_z.round(-2)
    df_input=df_input.groupby(["x","y","h_z"],as_index=False).mean()
    
    df_input_xy=df_input.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    df_est_xy=df_est.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    d_bool_list=cdist(df_est_xy.values,df_input_xy.values)
    d_bool_list = d_bool_list<=100000
    for i in tqdm(range(df_est_xy.shape[0])):
        input_list=[]
        input_xy = df_input_xy[d_bool_list[i,:]].values
        for x,y in input_xy:
            input_list.append(df_input[(df_input.x==x) & (df_input.y==y)])
        input_list = pd.concat(input_list).reset_index(drop=True)
        sv_all=[]
        for h_z in input_list.h_z.unique()[input_list.h_z.unique()<=0]:#
            pc_xy = input_list[input_list.h_z==h_z][["x","y"]].values
            pc_t = input_list[input_list.h_z==h_z][["t"]].values
            dist_xy = cdist(pc_xy,pc_xy)
            dist_t = cdist(pc_t,pc_t)**2
            sep=20000
            max_dist=70001
            sv_i = []
            for value in range(sep//2,max_dist,sep):
                res = dist_t[(dist_xy>=value) & (dist_xy<value+sep)]
                if res.size:
                    res = res.mean()
                else:
                    res = np.nan
                sv_i.append(res)
            sv_all.append(sv_i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sv_all =np.nanmean(np.vstack(sv_all),axis=0)
        model_sv.append(sv_all)
    model_sv = np.vstack(model_sv)
    model_sv = Variable(torch.from_numpy(model_sv).float()).to(device)
    return model_sv
model_sv = calc_model_semivariogram_xy(input_data,est_data)
# %%
# pd.DataFrame(model_sv.detach().cpu().numpy()).to_csv("./input/semivariogram/model_sv_extra.csv",index=False)
#%%
model_sv =pd.read_csv("./input/semivariogram/model_sv_extra.csv").values
model_sv= Variable(torch.from_numpy(model_sv).float())
model_sv.shape
model_sv = torch.cat((model_sv,model_sv),0)
model_sv.shape
#%%
np.nanmean(model_sv.detach().cpu().numpy(),axis=0)**2
#%%
sv_master = []
d_bool_list=cdist(est_data_origin_xy.values,input_data_origin_xy.values)
d_bool_list = d_bool_list<=100000
for i in tqdm(range(est_data_origin_xy.shape[0])):
    input_list=[]
    input_xy = input_data_origin_xy[d_bool_list[i,:]].values
    for x,y in input_xy:
        input_list.append(input_data[(input_data.x==x) & (input_data.y==y)])
    input_list = pd.concat(input_list).reset_index(drop=True)
    sv_all=[]
    for h_z in input_list.h_z.unique()[input_list.h_z.unique()<=0]:#
        pc_xy = input_list[input_list.h_z==h_z][["x","y"]].values
        pc_t = input_list[input_list.h_z==h_z][["t"]].values
        pc_xy_d = cdist(pc_xy,pc_xy)
        pc_t_d = cdist(pc_t,pc_t)
        sep=20000
        max_dist=70001
        sv_i = []
        for value in range(sep//2,max_dist,sep):
            res = pc_t_d[(pc_xy_d>=value) & (pc_xy_d<value+sep)]
            if res.size:
                res = res.mean()
            else:
                res = np.nan
            sv_i.append(res)
        sv_all.append(sv_i)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sv_all =np.nanmean(np.vstack(sv_all),axis=0)
    sv_master.append(sv_all)

# print(counter)
winsound.Beep(800,1000) 
# %%
np.vstack(res_list_all)
list(range(sep//2,max_dist,sep))
#%%
#%%
a=input_data_xy[d_bool_list[i,:]].values
a[np.random.choice(a.shape[0],20)].shape
#%%
est_data=pd.read_csv("./input/useful_data/est_grid_500.csv").sort_values(["x","y","h_z"],ascending=False)[["x","y","h_z"]]
est_data_xy=est_data.groupby(["x","y"],as_index=False).mean()[["x","y"]].values
dist_xy_list1=cdist(est_data_xy,est_data_xy)
est_data = Variable(torch.from_numpy(est_data.values).float())
est_data_xy = Variable(torch.from_numpy(est_data_xy).double())
dist_xy_list2 = torch.sqrt(pairwise_distances(est_data_xy))

temp=np.random.randint(0,100,(est_data.shape[0],1))
temp =Variable(torch.from_numpy(temp).float())
ts_est = torch.cat([est_data,temp],1).to(device)
z_unique = torch.unique(ts_est[:,2])
ts_est
#%%
dist_xy_list2
#%%
temp_dist_list=np.zeros((est_data_xy.shape[0],est_data_xy.shape[0]))
temp_dist_list=Variable(torch.from_numpy(temp_dist_list).float())
temp_dist_list
start_time = time.time()
est_bool = (dist_xy_list<=80000) & (dist_xy_list>0)
for i in range(est_data_xy.shape[0]):
    for j in np.where(est_bool[i,:])[0]:
        if temp_dist_list[i,j]==0:
            t_dist = (temp[i]-temp[j])**2
            temp_dist_list[i,j] = t_dist
            temp_dist_list[j,i] = t_dist
    sep=10000
    max_dist=30001
    res_list = []
    for value in range(sep//2,max_dist,sep):
        res = temp_dist_list[i,:][(dist_xy_list[i,:]>= value) & (dist_xy_list[i,:] < value+sep)]
        res = res.mean()/2
        res_list.append(res)
    # print(res_list)
print(time.time()-start_time)
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
def torch_select_column_by_condition(data, index, condition):
    condition_data = data[:, index]
    mask = condition_data.eq(condition)

    if len(torch.nonzero(mask)) == 0:
        return torch.Tensor()
    indices = torch.squeeze(torch.nonzero(mask), 1)
    select = torch.index_select(data, 0, indices)
    return select
#%%
start_time = time.time()
def calc_pred_semivariogram_xy(ts_est,dist_xy_list):
    sv_master = []
    for value in range(0,-5001,-1000):
        pc = torch_select_column_by_condition(ts_est,2,value)
        pred_t = pc[:,[3]]
        dist_t_list = pairwise_distances(pred_t,pred_t)
        sv_all = []
        for i in range(dist_xy_list.shape[0]):
            sep=20000
            max_dist=70001
            sv_i = []
            for value in range(sep//2,max_dist,sep*2):
                # value =10000
                res = dist_t_list[i,:][(dist_xy_list[i,:]>= value) & (dist_xy_list[i,:] < value+sep)]
                res = res.mean()
                sv_i.append(res)
            
            sv_i = torch.stack(sv_i)
            sv_all.append(sv_i)
        sv_all=torch.stack(sv_all)
        sv_master.append(sv_all)
    sv_master = torch.stack(sv_master).mean(axis=0)
    return sv_master
sv_master = calc_pred_semivariogram_xy(ts_est,dist_xy_list2)
print(time.time()-start_time)
sv_master
#%%
sv_master[~sv_master.isnan()].shape
#%%
start_time = time.time()
def calc_pred_semivariogram_xy(ts_est,dist_xy_list):
    sv_master = []
    for value in range(0,-5001,-1000):
        pc = torch_select_column_by_condition(ts_est,2,value)
        pred_t = pc[:,[3]]
        dist_t_list=Variable(torch.from_numpy(np.zeros((pc.shape[0],pc.shape[0]))).float()).to(device)
        dist_xy_bool = (dist_xy_list<=80000) & (dist_xy_list>0)
        for i in range(pc.shape[0]):
            for j in np.where(dist_xy_bool[i,:])[0]:
                if dist_t_list[i,j]==0:
                    t_dist = (pred_t[i]-pred_t[j])**2
                    dist_t_list[i,j] = t_dist
                    dist_t_list[j,i] = t_dist
        sv_all = []
        for i in range(dist_xy_list.shape[0]):
            sep=20000
            max_dist=70001
            sv_i = []
            for value in range(sep//2,max_dist,sep):
                res = dist_t_list[i,:][(dist_xy_list[i,:]>= value) & (dist_xy_list[i,:] < value+sep)]
                res = res.mean()
                sv_i.append(res)
            sv_i = torch.stack(sv_i)
            sv_all.append(sv_i)
        sv_all=torch.stack(sv_all)
        sv_master.append(sv_all)
    sv_master = torch.stack(sv_master).mean(axis=0)
    return sv_master
sv_master = calc_pred_semivariogram_xy(ts_est,dist_xy_list1)
print(time.time()-start_time)
sv_master
#%%
sv_master[~(model_sv[:,[0,2]].isnan() | sv_master.isnan())]
#%%
(model_sv[:,[0,2]].isnan() | sv_master.isnan())
#%%
diff_sv_xy =(sv_master-np.vstack(res_list_all))**2
diff_sv_xy[~diff_sv_xy.isnan()].mean()
#%%
plt.plot(np.sum(d_bool_list,axis=1),np.vstack(res_list_all)[:,0],".")
#%%
x=np.sum(d_bool_list,axis=1)
y=np.vstack(res_list_all)[:,0]
np.corrcoef(x[~np.isnan(y)],y[~np.isnan(y)])
# %%
3589/3633
# %%
np.vstack(res_list_all)[:,0]
# %%
wgs_to_albers=pd.read_csv("./input/WGS_to_albers/grid_WGS_albers.csv")
wgs_to_albers
# %%
est_data=pd.read_csv("./input/useful_data/est_grid_500.csv")
est_data_xy=est_data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
est_data_xy
# %%
est_data_xy["value"] = np.vstack(res_list_all)[:,0]
# %%
est_data_xy
# %%
idokeidovalue=wgs_to_albers.merge(est_data_xy,how="left",on=["x","y"])[["ido","keido","value"]]
idokeidovalue
# %%
x= idokeidovalue.keido
y= idokeidovalue.ido
t = idokeidovalue.value
sc = plt.scatter(x, y, vmin=0, c=t,marker='s',cmap=cm.jet,s=1)
plt.colorbar(sc)
plt.grid()
plt.show()
# %%
