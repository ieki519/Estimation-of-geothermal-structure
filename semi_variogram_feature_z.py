#%%
# %%
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
#%%
input_data=pd.read_csv("./input/useful_data/input_data.csv")
est_data=pd.read_csv("./input/useful_data/est_grid_500.csv")
est_data["t"]=np.random.randn(len(est_data))
input_data.h_z=input_data["h_z"].round(-2)
input_xy=input_data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
est_xy=est_data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
input_sv_all=[]
for x,y in input_xy.values:
    pc=input_data[(input_data.x==x)&(input_data.y==y)]
    h_z=pc[["h_z"]].values
    t=pc[["t"]].values
    dist_z=squareform(pdist(h_z))
    dist_t=squareform(pdist(t))
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
#%%
np.nanmean(input_sv_all,axis=0)
#%%
dist_nd=cdist(est_xy,input_xy)
mask = np.argsort(dist_nd[0,:])[:10]
mask
#%%
np.sort(dist_nd[0,:][mask])
#%%
np.sort(dist_nd[0,:])[:10]
#%%
model_sv_list=[]
for i in range(dist_nd.shape[0]):
    dist_nd_i = dist_nd[i,:][dist_nd[i,:]<=20000]
    mask=np.argsort(dist_nd_i)[:5]
    model_sv_list.append(np.nanmean(input_sv_all[mask,:],axis=0))
model_sv=np.vstack(model_sv_list)
model_sv
#%%
model_sv[:,[2]][~np.isnan(model_sv[:,[2]])].shape
#%%
np.unique(model_sv[:,[1]])[~np.isnan(np.unique(model_sv[:,[0]]))].shape
#%%
X_est=Variable(torch.from_numpy(est_data_origin.values).float())
XY_unique=Variable(torch.from_numpy(est_xy.values).float())
XY_unique
#%%
dist_z_list=[]
for x,y in XY_unique:
    pc=X_est[(X_est[:,0]==x) & (X_est[:,1]==y)]
    h_z=pc[:,[2]]
    dist_z=torch.sqrt(pairwise_distances(h_z))
    dist_z_list.append(dist_z)
    
sv_all=[]
for i,(x,y) in enumerate(XY_unique):
    pc=X_est[(X_est[:,0]==x) & (X_est[:,1]==y)]
    pred_t=pc[:,[3]]
    dist_z=dist_z_list[i]
    dist_t=pairwise_distances(pred_t)
    sv_i=[]
    for z in range(500,1501,500):
        sv_i.append(dist_t[dist_z==z].mean())
    sv_i=torch.stack(sv_i)
    sv_all.append(sv_i)
sv_all=torch.stack(sv_all)
#%%
model_sv_ts=Variable(torch.from_numpy(model_sv).float())

#%%
sv_loss=(model_sv_ts-sv_all)**2
sv_loss=sv_loss[~sv_loss.isnan()].mean()
sv_loss
#%%
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
        mask=dist_nd[i,:]<=80000
        model_sv_list.append(np.nanmean(sv_all[mask,:],axis=0))
    model_sv=np.vstack(model_sv_list)
    model_sv = Variable(torch.from_numpy(model_sv).float())
    return model_sv

model_sv=calc_model_semivariogram_z(input_data,est_data)

def create_d_z_list(ts_est,ts_est_xy):
    dist_z_list=[]
    for x,y in ts_est_xy:
        pc=ts_est[(ts_est[:,0]==x) & (ts_est[:,1]==y)]
        h_z=pc[:,[2]]
        dist_z=torch.sqrt(pairwise_distances(h_z))
        dist_z_list.append(dist_z)
    return dist_z_list
dist_z_list1 = create_d_z_list(X_est,XY_unique)

#%%
dist_z_list1[0]
#%%
def create_d_z_list(nd_est,nd_est_xy):
    dist_z_list=[]
    for x,y in nd_est_xy:
        pc=nd_est[(nd_est[:,0]==x) & (nd_est[:,1]==y)]
        h_z=pc[:,[2]]
        dist_z=squareform(pdist(h_z))
        dist_z_list.append(dist_z)
    return dist_z_list
dist_z_list2 = create_d_z_list(est_data_origin.values,est_xy.values)
#%%
dist_z_list1[0]==500
#%%
dist_z_list2 = Variable(torch.from_numpy(dist_z_list2).float())
#%%
for i in range(len(dist_z_list1)):
    # print(dist_z_list1[i].shape)
    # print(Variable(torch.from_numpy(dist_z_list2[i]).float()).shape)
    a = dist_z_list1[i]
    b = Variable(torch.from_numpy(dist_z_list2[i]).float())
    # print((a==b).all())
    if not (a==b).all():
        print(1)

#%%

start_time =time.time()
def calc_pred_semivariogram_z(ts_est,ts_est_xy,dist_z_list):
    sv_all=[]
    for i,(x,y) in enumerate(ts_est_xy):
        pc=ts_est[(ts_est[:,0]==x) & (ts_est[:,1]==y)]
        pred_t=pc[:,[3]]
        # dist_z = Variable(torch.from_numpy(dist_z_list[i]).float())
        dist_z=dist_z_list[i]
        dist_t=torch.sqrt(pairwise_distances(pred_t))
        sv_i=[]
        for z in range(500,1501,500):
            sv_i.append(dist_t[dist_z==z].mean())
        sv_i=torch.stack(sv_i)
        sv_all.append(sv_i)
    sv_all=torch.stack(sv_all)
    return sv_all
sv_all=calc_pred_semivariogram_z(X_est,XY_unique,dist_z_list1)
print(time.time()-start_time)
sv_loss=(model_sv-sv_all)**2
sv_loss=sv_loss[~sv_loss.isnan()].mean()
sv_loss
#%%

#%%
def calc_model_semivariogram_z(df,df_xy):
    sv_all=[]
    for x,y in df_xy.values:
        pc=df[(df.x==x)&(df.y==y)]
        h_z=pc[["h_z"]].values
        t=pc[["t"]].values
        dist_z=squareform(pdist(h_z))
        dist_t=squareform(pdist(t))
        sv_i=np.zeros(len(range(0,1501,500)[1:]))
        for i,dist in enumerate(range(0,1501,500)[1:]):
            mask=dist_z==dist
            res=dist_t[mask]
            if res.size:
                sv_i[i]=res.mean()
            else:
                sv_i[i]=np.nan
        sv_all.append(sv_i)
    sv_all=np.vstack(sv_all)
    sv_all=np.nanmean(sv_all,axis=0)
    return sv_all
#%%
sv_list=[]
for condition in mask_list:
    sv=calc_model_semivariogram_z(input_data,input_xy[condition])
    sv_list.append(sv)
#%%
np.array(sv_list)
# %%
plt.plot(sv_all)
# %%
list()
#%%
for i in range(0,5001,500)[1:]:
    print(i)
# %%
len(range(0,5001,500)[1:])
# %%
print([].append([]))
# %%
dict_t=[[]]*len(range(0,5001,500)[1:])
dict_t
# %%
dict_t[0]+=[1,2,3,4]
dict_t
# %%
dict_t=np.array([[]]*len(range(0,5001,500)[1:]))
np.append(dict_t[0],[1,2,3,4])
dict_t
# %%
