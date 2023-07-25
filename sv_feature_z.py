#%%
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
# %%
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

def extra_split(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.1, shuffle=False)
    return train_data, test_data
# %%
master_features=['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
                    'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
                        'group_rank', 'age']

# preprocess
input_data = pd.read_csv('./input/useful_data/input_data.csv')
input_data = preprocess_input(input_data)

train_data, test_data = extra_split(input_data)

est_data = pd.read_csv('./input/useful_data/est_grid_500.csv').sort_values(["x","y","h_z"])
curie_data = pd.read_csv('./input/curie_point/grid_curie.csv')
curie_data_580ika = pd.read_csv('./input/curie_point/grid_curie_580ika.csv')
curie_data_580izyou = pd.read_csv('./input/curie_point/grid_curie_580izyou.csv')

est_data = preprocess_grid(est_data)
curie_data = preprocess_grid(curie_data)
curie_data_580ika = preprocess_grid(curie_data_580ika)
curie_data_580izyou = preprocess_grid(curie_data_580izyou)

# depth feature 
depth_target_list=[0,500,800]
depth_features = master_features.copy()

train_data_plane = train_data.groupby(['x', 'y'], as_index=False).mean()
test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

for depth_target in depth_target_list:
    train_data_depth = train_data[train_data.h_z==-depth_target].copy()
    
    depth_target=f"depth{depth_target}"
    train_data_depth[depth_target] =train_data_depth["t"]
    
    X_depth = train_data_depth[depth_features+[depth_target]]
    X_train = train_data_plane[depth_features]
    X_test = test_data_plane[depth_features]
    X_est = est_data_plane[depth_features]

    stacking_model=Stacking_train_model(X_depth,depth_target)
    train_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_train)
    test_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_test)
    est_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_est)
    
    train_data = train_data.merge(train_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
    test_data = test_data.merge(test_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
    est_data = est_data.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
    curie_data=curie_data.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
    curie_data_580ika=curie_data_580ika.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
    curie_data_580izyou=curie_data_580izyou.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
    master_features += [depth_target]

# grad feature
train_data = grad_calc(train_data)
train_data=grad_maxmin_calc(train_data)

train_data_plane = train_data.groupby(
    ['x', 'y'], as_index=False).mean()
test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

grad_features = master_features.copy()
grad_target_list = ['grad','grad_max','grad_min']

for grad_target in grad_target_list:
    X_train = train_data_plane[grad_features+[grad_target]]
    X_test = test_data_plane[grad_features]
    X_est = est_data_plane[grad_features]

    stacking_model=Stacking_train_model(X_train,grad_target)
    test_data_plane[grad_target] = Stacking_est(stacking_model,X_train,grad_target,X_test)
    est_data_plane[grad_target] = Stacking_est(stacking_model,X_train,grad_target,X_est)
    
    test_data = test_data.merge(test_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
    est_data = est_data.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
    curie_data=curie_data.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
    curie_data_580ika=curie_data_580ika.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
    curie_data_580izyou=curie_data_580izyou.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
    master_features += [grad_target]
#%%
train_data
#%%
def calc_model_semivariogram_z(df_input):
    df_input = df_input.copy()
    df_input.h_z = df_input.h_z.round(-2)
    df_input_xy=df_input.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    sv_all=[]
    for x,y in df_input_xy.values:
        pc=df_input[(df_input.x==x)&(df_input.y==y)]
        h_z=pc[["h_z"]].values
        t=pc[["t"]].values
        dist_z=squareform(pdist(h_z))
        dist_t=squareform(pdist(t))**2
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
    sv_all=pd.DataFrame(sv_all,columns=["sv_500","sv_1000","sv_1500"])
    df_input_xy = pd.concat([df_input_xy,sv_all],axis=1)
    return df_input_xy
# %%
train_sv_plane =calc_model_semivariogram_z(train_data)

train_data_plane = train_data.groupby(
    ['x', 'y'], as_index=False).mean()
train_data_plane = train_data_plane.merge(train_sv_plane,on =["x","y"],how="left")
test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

sv_features = master_features.copy()
sv_target_list = ['sv_500','sv_1000']
for sv_target in sv_target_list:
    train_data_plane_is = train_data_plane[~train_data_plane[sv_target].isnull()]
    train_data_plane_isnull = train_data_plane[train_data_plane[sv_target].isnull()]
    train_data_plane_isnull = train_data_plane_isnull.drop(columns=sv_target)

    X_train = train_data_plane_is[sv_features+[sv_target]]
    X_train_isnull = train_data_plane_isnull[sv_features]
    X_test = test_data_plane[sv_features]
    X_est = est_data_plane[sv_features]

    stacking_model=Stacking_train_model(X_train,sv_target)
    train_data_plane_isnull[sv_target] = Stacking_est(stacking_model,X_train,sv_target,X_train_isnull)
    test_data_plane[sv_target] = Stacking_est(stacking_model,X_train,sv_target,X_test)
    est_data_plane[sv_target] = Stacking_est(stacking_model,X_train,sv_target,X_est)

    train_data_plane = pd.concat([train_data_plane_is,train_data_plane_isnull],axis=0)

    train_data = train_data.merge(train_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
    test_data = test_data.merge(test_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
    est_data = est_data.merge(est_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
    curie_data=curie_data.merge(est_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
    curie_data_580ika=curie_data_580ika.merge(est_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
    curie_data_580izyou=curie_data_580izyou.merge(est_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
    master_features += [sv_target]
# %%
model_sv_z = est_data.groupby(["x","y"],as_index=False).mean()[sv_target_list].values
model_sv_z = Variable(torch.from_numpy(model_sv_z).float()).to(device)
model_sv_z
# %%
model_sv_z.shape
# %%
train_data.shape[0]+test_data.shape[0]
# %%
train_data
# %%
