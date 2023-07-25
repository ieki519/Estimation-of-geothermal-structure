#%%
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
import folium
torch.backends.cudnn.benchmark = True
import matplotlib.cm as cm
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# %%
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

# def preprocess_grid(df):
#     add_grid_volcano = pd.read_csv('./input/volcano/add_grid_volcano.csv')
#     add_grid_curie = pd.read_csv('./input/curie_point/add_grid_curie.csv')
#     add_grid_onsen = pd.read_csv('./input/onsen/add_grid_onsen.csv')

#     df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
#     df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
#     df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])

#     return df

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
        h = df[(df['x'] == x) & (
            df['y'] == y)][['h']].values.max()
        zt.z = zt.z.round(-2)
        zt = zt.groupby("z",as_index=False).mean()
        if zt.shape[0] == 1:
            zt = pd.DataFrame(np.array([[0, 0]]), columns=[
                              'z', 't']).append(zt)
        zt = zt.sort_values('z')
        z_diff = np.diff(zt['z'])
        t_diff = np.diff(zt['t'])
        hz = h-zt[:-1].z.values-z_diff/2
        
        grad = (t_diff/z_diff)*1000
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_max'] = max(grad)
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_min'] = min(grad)
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_max_h_z'] = hz[np.argmax(grad)]
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_min_h_z'] = hz[np.argmin(grad)]
    return df

def feature_engineer(df):
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
def Stacking_model(df_train_s,target_s,df_est_s):
    trn_s=df_train_s.drop(target_s,axis=1).reset_index(drop=True)
    tst_s=df_train_s[target_s].reset_index(drop=True)
    train_features_s=pd.DataFrame(np.zeros((trn_s.shape[0],3)))
    cv_s = KFold(n_splits=5, shuffle=True, random_state=0)
    est_features_s_list =[]
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
        
        est_features_s=pd.DataFrame(np.zeros((df_est_s.shape[0],3)))
        est_features_s[0]=rf.predict(df_est_s)
        est_features_s[1]=lgbm.predict(df_est_s)
        est_features_s[2]=xgb.predict(df_est_s)
        est_features_s_list.append(est_features_s.values)
        
    est_features_s=np.mean(est_features_s_list,axis=0)
    lr=LinearRegression()
    lr.fit(train_features_s,tst_s)
    pred_s = lr.predict(est_features_s)
    return pred_s

def Stacking_model_ok(df_train_s,target_s,df_est_s,ok_parameter):
    trn_s=df_train_s.drop(target_s,axis=1).reset_index(drop=True)
    tst_s=df_train_s[target_s].reset_index(drop=True)
    train_features_s=pd.DataFrame(np.zeros((trn_s.shape[0],4)))
    cv_s = KFold(n_splits=5, shuffle=True, random_state=0)
    est_features_s_list =[]
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
        ok = OrdinaryKriging(trn_x_s['x'], trn_x_s['y'], trn_y_s, 
                variogram_parameters=ok_parameter[0],variogram_model=ok_parameter[1])

        train_features_s.iloc[val_idx_s, 0]=rf.predict(val_x_s)
        train_features_s.iloc[val_idx_s, 1]=lgbm.predict(val_x_s)
        train_features_s.iloc[val_idx_s, 2]=xgb.predict(val_x_s)
        train_features_s.iloc[val_idx_s, 3]=ok.execute('points', val_x_s['x'], val_x_s['y'])[0].data
        
        est_features_s=pd.DataFrame(np.zeros((df_est_s.shape[0],4)))
        est_features_s[0]=rf.predict(df_est_s)
        est_features_s[1]=lgbm.predict(df_est_s)
        est_features_s[2]=xgb.predict(df_est_s)
        est_features_s[3]=ok.execute('points', df_est_s['x'], df_est_s['y'])[0].data
        
        est_features_s_list.append(est_features_s.values)
        
    est_features_s=np.mean(est_features_s_list,axis=0)
    lr=LinearRegression()
    lr.fit(train_features_s,tst_s)
    pred_s = lr.predict(est_features_s)
    return pred_s

def Stacking_train_model_ok(df_train_s,target_s,ok_parameter):
    trn_s=df_train_s.drop(target_s,axis=1).reset_index(drop=True)
    tst_s=df_train_s[target_s].reset_index(drop=True)
    train_features_s=pd.DataFrame(np.zeros((trn_s.shape[0],4)))
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
        ok = OrdinaryKriging(trn_x_s['x'], trn_x_s['y'], trn_y_s, 
                variogram_parameters=ok_parameter[0],variogram_model=ok_parameter[1])

        train_features_s.iloc[val_idx_s, 0]=rf.predict(val_x_s)
        train_features_s.iloc[val_idx_s, 1]=lgbm.predict(val_x_s)
        train_features_s.iloc[val_idx_s, 2]=xgb.predict(val_x_s)
        train_features_s.iloc[val_idx_s, 3]=ok.execute('points', val_x_s['x'], val_x_s['y'])[0].data

    lr=LinearRegression()
    lr.fit(train_features_s,tst_s)
    return lr

def Stacking_est_ok(model_s,df_train_s,target_s,df_est_s,ok_parameter):
    trn_s=df_train_s.drop(target_s,axis=1)
    tst_s=df_train_s[target_s]

    rf=RandomForestRegressor(random_state=0)
    lgbm=LGBMRegressor(random_state=0)
    xgb=XGBRegressor(random_state=0)

    rf.fit(trn_s, tst_s)
    lgbm.fit(trn_s, tst_s)
    xgb.fit(trn_s, tst_s)
    ok = OrdinaryKriging(trn_s['x'], trn_s['y'], tst_s, 
                variogram_parameters=ok_parameter[0],variogram_model=ok_parameter[1])

    est_features_s=pd.DataFrame(np.zeros((df_est_s.shape[0],4)))
    est_features_s[0]=rf.predict(df_est_s)
    est_features_s[1]=lgbm.predict(df_est_s)
    est_features_s[2]=xgb.predict(df_est_s)
    est_features_s[3]=ok.execute('points', df_est_s['x'], df_est_s['y'])[0].data

    pred_s=model_s.predict(est_features_s)
    return pred_s

def extra_split(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.1, shuffle=False)
    return train_data, test_data

def extra_split80(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
    return train_data, test_data
#%%
# %%
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data=preprocess_input(input_data)
# input_data=feature_engineer(input_data)
# input_data = grad_calc(input_data)
# input_data = grad_maxmin_calc(input_data)
input_data.h_z=input_data.h_z.round(-1)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
input_data.columns
#%%
input_data_0=input_data[input_data.h_z==0]
input_data_500=input_data[input_data.h_z==-500]
input_data_800=input_data[input_data.h_z==-800]
input_data_1000=input_data[input_data.h_z==-1000]
input_data_1500=input_data[input_data.h_z==-1500]
input_data_1000
#%%
def variogram(df, sep=11000, max_dist=500001, parameters={'sill': 9, 'range': 200000, 'nugget': 0.5}, how='gaussian'):
    xy_dis = squareform(pdist(df.iloc[:, [0, 1]]))
    t_vario = squareform(pdist(df.iloc[:, 2].values.reshape(-1, 1))**2)
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
    plt.plot(x[:], sv_i[:], c='black', marker='o')
    if how == 'gaussian':
        plt.plot(x[:], vm.gaussian_variogram_model(
            [sill_, range_, nugget_], np.array(x))[:], c='red')
    elif how == 'spherical':
        plt.plot(x[:], vm.spherical_variogram_model(
            [sill_, range_, nugget_], np.array(x))[:], c='red')
    print(df.columns[2])
    plt.show()
#%%
# inter
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data=preprocess_input(input_data)
input_data.h_z=input_data.h_z.round(-1)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
xy_data = input_data.groupby(
        ['x', 'y'], as_index=False).mean().loc[:, ['x', 'y']]
cv = KFold(n_splits=5, shuffle=True, random_state=0)
for num, (idx_trn, idx_tst) in enumerate(cv.split(xy_data)):
    est_data = pd.read_csv('./input_japan/useful_data/est_grid_detail_ja.csv')
    est_data = preprocess_grid(est_data)
    
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
    
    train_data.h_z=train_data.h_z.round(-1)
    train_data=train_data.groupby(["x","y","h_z"],as_index=False).mean()

    train_data_0=train_data[train_data.h_z==0]
    train_data_500=train_data[train_data.h_z==-500]
    train_data_800=train_data[train_data.h_z==-800]
    train_data_1000=train_data[train_data.h_z==-1000]
    train_data_1500=train_data[train_data.h_z==-1500]
    print()
    variogram(train_data_0[["x","y","t"]], sep=1100, max_dist=20001, parameters={
            'sill': 2500-500, 'range': 6000, 'nugget': 500}, how='spherical')
    variogram(train_data_500[["x","y","t"]], sep=1100, max_dist=20001, parameters={
            'sill': 3000-500, 'range': 5000, 'nugget': 500}, how='spherical')
    variogram(train_data_800[["x","y","t"]], sep=1100, max_dist=20001, parameters={
            'sill': 2000-250, 'range': 3000, 'nugget': 250}, how='spherical')
    # variogram(train_data_1000[["x","y","t"]], sep=1100, max_dist=20001, parameters={
    #         'sill': 3000-1250, 'range': 20000, 'nugget': 1250}, how='spherical')
    variogram(train_data_1000[["x","y","t"]], sep=5000, max_dist=100001, parameters={
            'sill': 3000-1250, 'range': 20000, 'nugget': 1250}, how='gaussian')#3000-1250, 'range': 20000, 'nugget': 1250
#%%
input_data_1000=train_data[train_data.h_z==-1000]
variogram(input_data_1000[["x","y","t"]], sep=1000, max_dist=20001, parameters={
            'sill': 3000-500, 'range': 4000, 'nugget': 500}, how='gaussian')
#%%
parameters = {
    0: [{'sill': 2500-500, 'range': 6000, 'nugget': 500}, 'spherical'],
    500: [{'sill': 3000-500, 'range': 5000, 'nugget': 500}, "spherical"],
    800: [{'sill': 2000-250, 'range': 3000, 'nugget': 250}, 'spherical'],
    1000: [{'sill': 3000-1250, 'range': 20000, 'nugget': 1250}, 'gaussian'],#'sill': 3000-1250, 'range': 20000, 'nugget': 1250
}
#%%
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data=preprocess_input(input_data)
input_data.h_z=input_data.h_z.round(-1)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
features =['x', 'y', 'h', 'volcano', 'curie',
       'age_a', 'age_b', 'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4',
       'HCO3', 'anion', 'group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩',
       'group_ja_変成岩', 'group_ja_火成岩', 'age']#+['grad','grad_max','grad_min','grad_max_h_z','grad_min_h_z']
result_df=[]
for key in [1000]:#[0,500,800,1000]:
    fix_seed(0)
    result_dict={}
    target="t"
    trn=input_data[input_data.h_z==-key][features].reset_index(drop=True)
    tst=input_data[input_data.h_z==-key][target].reset_index(drop=True)
    print(f'{key}', '-'*10, tst.std())
    
    MLA = {'xgb': XGBRegressor(random_state=0), 'lgbm': LGBMRegressor(random_state=0), 'rf': RandomForestRegressor(random_state=0)}
    for m in MLA.keys():
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        result = model_selection.cross_validate(
            MLA[m], trn, tst, cv=cv, scoring='neg_mean_squared_error')
        print(m, np.sqrt(-result['test_score'].mean()))
        result_dict[m]=np.sqrt(-result['test_score'].mean())
        
    rmse_list1 = []
    rmse_list2 = []
    mape_list =[]
    for trn_idx, val_idx in cv.split(trn, tst):
        trn_x = trn.iloc[trn_idx, :]
        trn_y = tst[trn_idx]

        val_x = trn.iloc[val_idx, :]
        val_y = tst[val_idx]

        ok1 = OrdinaryKriging(trn_x['x'], trn_x['y'], trn_y, variogram_model=parameters[key]
                             [1], variogram_parameters=parameters[key][0])
        # ok2 = OrdinaryKriging(trn_x['x'], trn_x['y'], trn_y)
        predict1 = ok1.execute('points', val_x['x'], val_x['y'])[0].data
        # predict2 = ok2.execute('points', val_x['x'], val_x['y'])[0].data
        rmse_list1.append(np.sqrt(mean_squared_error(val_y.values, predict1)))
        mape_list.append(np.mean(np.abs(val_y.values-predict1)/val_y.values)*100)
        
        # rmse_list2.append(np.sqrt(mean_squared_error(val_y.values, predict2)))
        
    print('ok1', np.mean(rmse_list1))
    print("mape",np.mean(mape_list))
    # print('ok2', np.mean(rmse_list2))
    result_dict['ok1']=np.mean(rmse_list1)
    # result_dict['ok2']=np.mean(rmse_list2)
    
    start_time = time.time()
    rmse_list=[]
    trn=input_data[input_data.h_z==-key][features].reset_index(drop=True)
    tst=input_data[input_data.h_z==-key][target].reset_index(drop=True)
    trn=pd.concat([trn,tst],axis=1)
    for trn_idx, val_idx in cv.split(trn):
        trn_x = trn.iloc[trn_idx, :]
        val_x = trn.iloc[val_idx, :]
        # model_s=Stacking_train_model(trn_x,target)
        # pred_s=Stacking_est(model_s,trn_x,target,val_x.drop(target,axis=1))
        pred_s=Stacking_model(trn_x,target,val_x.drop(target,axis=1))
        rmse_list.append(np.sqrt(mean_squared_error(val_x[target].values,pred_s)))
    print("st",np.mean(rmse_list),time.time()-start_time)
    result_dict['st']=np.mean(rmse_list)
    
    start_time = time.time()
    rmse_list=[]
    trn=input_data[input_data.h_z==-key][features].reset_index(drop=True)
    tst=input_data[input_data.h_z==-key][target].reset_index(drop=True)
    trn=pd.concat([trn,tst],axis=1)
    for trn_idx, val_idx in cv.split(trn):
        trn_x = trn.iloc[trn_idx, :]
        val_x = trn.iloc[val_idx, :]
        # model_s=Stacking_train_model_ok(trn_x,target,parameters[key])
        # pred_s=Stacking_est_ok(model_s,trn_x,target,val_x.drop(target,axis=1),parameters[key])
        pred_s=Stacking_model_ok(trn_x,target,val_x.drop(target,axis=1),parameters[key])
        rmse_list.append(np.sqrt(mean_squared_error(val_x[target].values,pred_s)))
    print("st_ok",np.mean(rmse_list),time.time()-start_time)
    result_dict['st_ok']=np.mean(rmse_list)
    
    result_dict=pd.DataFrame(result_dict.values(),index=result_dict.keys(),columns=[key])
    # print(result_dict)
    result_df.append(result_dict)
result_df=pd.concat(result_df,axis=1)
#%%
result_df
#%%
# result_df.to_csv("./input_japan/depth/database/depth_pred_result.csv")
#%%
# albers_wgs=pd.read_csv("./input/WGS_to_albers/input_WGS_albers.csv")
# albers_wgs.x,albers_wgs.y=albers_wgs.x.round(),albers_wgs.y.round()
# xy=input_data_1000.merge(albers_wgs,on=["x","y"],how="left")[["ido","keido"]]
# #%%
# def visualize_locations(df,  zoom=4):
#     """日本を拡大した地図に、pandasデータフレームのlatitudeおよびlongitudeカラムをプロットする。
#     """
        	
#     # 図の大きさを指定する。
#     f = folium.Figure(width=1000, height=500)

#     # 初期表示の中心の座標を指定して地図を作成する。
#     center_lat=34.686567
#     center_lon=135.52000
#     m = folium.Map([center_lat,center_lon], zoom_start=zoom).add_to(f)
        
#     # データフレームの全ての行のマーカーを作成する。
#     for i in range(0,len(df)):
#         folium.CircleMarker(location=[df["ido"][i],df["keido"][i]],radius=1,fill=True).add_to(m)
        
#     return m
# #%%
# visualize_locations(xy)
# %%
input_data_0.columns
#%%
parameters = {
    "depth0": [{'sill': 2500-500, 'range': 6000, 'nugget': 500}, 'spherical'],
    "depth500": [{'sill': 3000-500, 'range': 5000, 'nugget': 500}, "spherical"],
    "depth800": [{'sill': 2000-250, 'range': 3000, 'nugget': 250}, 'spherical'],
    "depth1000": [{'sill': 3000-1250, 'range': 20000, 'nugget': 1250}, 'gaussian'],#'sill': 3000-1250, 'range': 20000, 'nugget': 1250
}
depth_features = ['x', 'y', 'h', 'volcano', 'curie',
       'age_a', 'age_b', 'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4',
       'HCO3', 'anion', 'group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩',
       'group_ja_変成岩', 'group_ja_火成岩', 'age']
#%%
fix_seed(0)
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)
input_data_plane = input_data.groupby(
    ['x', 'y'], as_index=False).mean()

est_data = pd.read_csv('./input_japan/useful_data/est_grid_detail_ja.csv')
est_data = preprocess_grid(est_data)
est_data_plane=est_data.groupby(
    ['x', 'y'], as_index=False).mean()
#%%
train_data = input_data.copy()
train_data.h_z=train_data.h_z.round(-1)
train_data=train_data.groupby(["x","y","h_z"],as_index=False).mean()
#%%
for depth_target in [0,500,800]:
    X_train=train_data[train_data.h_z==-depth_target][depth_features+["t"]].reset_index(drop=True)
    X_input=input_data_plane[depth_features]
    X_est = est_data_plane[depth_features].astype(float)
    # stacking_model=Stacking_train_model_ok(X_train,"t",parameters[f"depth{depth_target}"])
    # input_data_plane[f"depth{depth_target}"] = Stacking_est_ok(stacking_model,X_train,"t",X_input,parameters[f"depth{depth_target}"])
    # est_data_plane[f"depth{depth_target}"] = Stacking_est_ok(stacking_model,X_train,"t",X_est,parameters[f"depth{depth_target}"])
    input_data_plane[f"depth{depth_target}"] = Stacking_model_ok(X_train,"t",X_input,parameters[f"depth{depth_target}"])
    est_data_plane[f"depth{depth_target}"] = Stacking_model_ok(X_train,"t",X_est,parameters[f"depth{depth_target}"])
    
for depth_target in [1000]:
    X_train=train_data[train_data.h_z==-depth_target][depth_features+["t"]].reset_index(drop=True)
    X_input=input_data_plane[depth_features]
    X_est = est_data_plane[depth_features].astype(float)
    # stacking_model=Stacking_train_model_ok(X_train,"t",parameters[f"depth{depth_target}"])
    # input_data_plane[f"depth{depth_target}"] = Stacking_est_ok(stacking_model,X_train,"t",X_input,parameters[f"depth{depth_target}"])
    # est_data_plane[f"depth{depth_target}"] = Stacking_est_ok(stacking_model,X_train,"t",X_est,parameters[f"depth{depth_target}"])
    input_data_plane[f"depth{depth_target}"] = Stacking_model(X_train,"t",X_input)
    est_data_plane[f"depth{depth_target}"] = Stacking_model(X_train,"t",X_est)
est_data_plane=est_data_plane[["x","y"]+["depth0","depth500","depth800","depth1000"]]
est_data_plane.to_csv("./input_japan/depth/add_grid_depth_detail_ja.csv",index=False)
input_data_plane=input_data_plane[["x","y"]+["depth0","depth500","depth800","depth1000"]]
input_data_plane.to_csv("./input_japan/depth/add_input_depth_ja.csv",index=False)
#%%
# inter
fix_seed(0)
cv = KFold(n_splits=5, shuffle=True, random_state=0)
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)
xy_data = input_data.groupby(
        ['x', 'y'], as_index=False).mean().loc[:, ['x', 'y']]

for num, (idx_trn, idx_tst) in enumerate(cv.split(xy_data)):
    fix_seed(0)
    est_data = pd.read_csv('./input_japan/useful_data/est_grid_detail_ja.csv')
    est_data = preprocess_grid(est_data)
    
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

    train_data_d=train_data.copy()
    train_data_d.h_z=train_data_d.h_z.round(-1)
    train_data_d=train_data_d.groupby(["x","y","h_z"],as_index=False).mean()
    
    train_data_plane = train_data.groupby(
        ['x', 'y'], as_index=False).mean()
    test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
    est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

    for depth_target in [0,500,800]:
        train_data_depth = train_data_d[train_data_d.h_z==-depth_target].copy()
        
        depth_target=f"depth{depth_target}"
        train_data_depth[depth_target] =train_data_depth["t"]
        
        X_depth = train_data_depth[depth_features+[depth_target]]
        X_train = train_data_plane[depth_features]
        X_test = test_data_plane[depth_features]
        X_est = est_data_plane[depth_features].astype(float)

        # stacking_model=Stacking_train_model_ok(X_depth,depth_target,parameters[depth_target])
        # train_data_plane[depth_target] = Stacking_est_ok(stacking_model,X_depth,depth_target,X_train,parameters[depth_target])
        # test_data_plane[depth_target] = Stacking_est_ok(stacking_model,X_depth,depth_target,X_test,parameters[depth_target])
        # est_data_plane[depth_target] = Stacking_est_ok(stacking_model,X_depth,depth_target,X_est,parameters[depth_target])
        train_data_plane[depth_target] = Stacking_model_ok(X_depth,depth_target,X_train,parameters[depth_target])
        test_data_plane[depth_target] = Stacking_model_ok(X_depth,depth_target,X_test,parameters[depth_target])
        est_data_plane[depth_target] = Stacking_model_ok(X_depth,depth_target,X_est,parameters[depth_target])
        
    for depth_target in [1000]:
        train_data_depth = train_data[train_data.h_z==-depth_target].copy()
        
        depth_target=f"depth{depth_target}"
        train_data_depth[depth_target] =train_data_depth["t"]
        
        X_depth = train_data_depth[depth_features+[depth_target]]
        X_train = train_data_plane[depth_features]
        X_test = test_data_plane[depth_features]
        X_est = est_data_plane[depth_features]

        # stacking_model=Stacking_train_model(X_depth,depth_target)
        # train_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_train)
        # test_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_test)
        # est_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_est)
        train_data_plane[depth_target] = Stacking_model(X_depth,depth_target,X_train)
        test_data_plane[depth_target] = Stacking_model(X_depth,depth_target,X_test)
        est_data_plane[depth_target] = Stacking_model(X_depth,depth_target,X_est)
        
    df_input = pd.concat([train_data_plane,test_data_plane])
    df_input=df_input[["x","y","depth0","depth500","depth1000"]]
    est_data_plane=est_data_plane[["x","y","depth0","depth500","depth1000"]]
    df_input.to_csv(f"./input_japan/depth/add_input_depth_ja_inter_{num}.csv",index=False)
    est_data_plane.to_csv(f"./input_japan/depth/add_grid_depth_detail_ja_inter_{num}.csv",index=False)
#%%
train_data_depth
#%%
# extra
fix_seed(0)
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)

train_data, test_data = extra_split(input_data)

est_data = pd.read_csv('./input_japan/useful_data/est_grid_500_ja.csv')
est_data = preprocess_grid(est_data)


train_data_d=train_data.copy()
train_data_d.h_z=train_data_d.h_z.round(-1)
train_data_d=train_data_d.groupby(["x","y","h_z"],as_index=False).mean()
    
train_data_plane = train_data.groupby(['x', 'y'], as_index=False).mean()
test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

for depth_target in [0,500,800]:
    train_data_depth = train_data_d[train_data_d.h_z==-depth_target].copy()
    
    depth_target=f"depth{depth_target}"
    train_data_depth[depth_target] =train_data_depth["t"]
    
    X_depth = train_data_depth[depth_features+[depth_target]]
    X_train = train_data_plane[depth_features]
    X_test = test_data_plane[depth_features]
    X_est = est_data_plane[depth_features].astype(float)

    # stacking_model=Stacking_train_model_ok(X_depth,depth_target,parameters[depth_target])
    # train_data_plane[depth_target] = Stacking_est_ok(stacking_model,X_depth,depth_target,X_train,parameters[depth_target])
    # test_data_plane[depth_target] = Stacking_est_ok(stacking_model,X_depth,depth_target,X_test,parameters[depth_target])
    # est_data_plane[depth_target] = Stacking_est_ok(stacking_model,X_depth,depth_target,X_est,parameters[depth_target])
    
    train_data_plane[depth_target] = Stacking_model_ok(X_depth,depth_target,X_train,parameters[depth_target])
    test_data_plane[depth_target] = Stacking_model_ok(X_depth,depth_target,X_test,parameters[depth_target])
    est_data_plane[depth_target] = Stacking_model_ok(X_depth,depth_target,X_est,parameters[depth_target])
    
df_input = pd.concat([train_data_plane,test_data_plane])
df_input=df_input[["x","y","depth0","depth500","depth800"]].groupby(["x","y"],as_index=False).mean()
est_data_plane=est_data_plane[["x","y","depth0","depth500","depth800"]]
df_input.to_csv(f"./input_japan/depth/add_input_depth_ja_extra_0.csv",index=False)
est_data_plane.to_csv(f"./input_japan/depth/add_grid_depth_detail_ja_extra_0.csv",index=False)
#%%
df_input
#%%
test_data_plane
#%%
a=pd.concat([train_data_plane[depth_features],test_data_plane[depth_features]]).reset_index(drop=True)
a.round().drop_duplicates()
#%%
# df_input.to_csv(f"./input_japan/grad/add_input_grad_ja_extra.csv",index=False)
# est_data_plane.to_csv(f"./input_japan/grad/add_grid_grad_ja_extra.csv")



#%%
df=input_data_500.copy()

result_dict={}
trn = df.drop(columns="t")
tst = df["t"]

MLA = {'xgb': XGBRegressor(), 'lgbm': LGBMRegressor(),  'rf': RandomForestRegressor()}
for m in MLA.keys():
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    result = model_selection.cross_validate(
        MLA[m], trn, tst, cv=cv, scoring='neg_mean_squared_error')
    print(m, np.sqrt(-result['test_score'].mean()))
    result_dict[m]=np.sqrt(-result['test_score'].mean())
    
rmse_list1 = []
rmse_list2 = []

for trn_idx, val_idx in cv.split(trn, tst):
    trn_x = trn.iloc[trn_idx, :]
    trn_y = tst.iloc[trn_idx]

    val_x = trn.iloc[val_idx, :]
    val_y = tst.iloc[val_idx]

    ok1 = OrdinaryKriging(trn_x['x'], trn_x['y'], trn_y, variogram_model="spherical"
                            , variogram_parameters=[3000-500,5000,500])
    # 0:[2500-1000,8000,1000]
    # 500:[3000-1250,5000,1250]
    # 0[2200-200,4000,200]
    # 500 [3000-500,5000,500]
    ok2 = OrdinaryKriging(trn_x['x'], trn_x['y'], trn_y)
    predict1 = ok1.execute('points', val_x['x'], val_x['y'])[0].data
    predict2 = ok2.execute('points', val_x['x'], val_x['y'])[0].data
    rmse_list1.append(np.sqrt(mean_squared_error(val_y.values, predict1)))
    rmse_list2.append(np.sqrt(mean_squared_error(val_y.values, predict2)))

print('ok1', np.mean(rmse_list1))
print('ok2', np.mean(rmse_list2))
result_dict['ok1']=np.mean(rmse_list1)
result_dict['ok2']=np.mean(rmse_list2)

# Voting
vote_est = [
    ('rf0', RandomForestRegressor(random_state=0)),
    ('lgbm0', LGBMRegressor(random_state=0)),
    ('xgb0', XGBRegressor(random_state=0)),
    ]
vr=VotingRegressor(estimators=vote_est)
cv = KFold(n_splits=5, shuffle=True, random_state=0)
result = model_selection.cross_validate(
    vr, trn, tst, cv=cv, scoring='neg_mean_squared_error')
print('vr', np.sqrt(-result['test_score'].mean()))
result_dict['vr']=np.sqrt(-result['test_score'].mean())

# Stacking
rmse_list=[]
trn["t"]=tst
for trn_idx, val_idx in cv.split(trn):
    trn_x = trn.iloc[trn_idx, :]
    val_x = trn.iloc[val_idx, :]
    model_s=Stacking_train_model(trn_x,"t")
    pred_s=Stacking_est(model_s,trn_x,"t",val_x.drop("t",axis=1))
    rmse_list.append(np.sqrt(mean_squared_error(val_x["t"].values,pred_s)))
print("st",np.mean(rmse_list))
result_dict['stacking']=np.mean(rmse_list)

result_dict=pd.DataFrame(result_dict.values(),index=result_dict.keys(),columns=["t"])
# %%


#%%
xy_dis=squareform(pdist(input_data_0[['x','y']]))
t_vario=squareform(pdist(input_data_0['t'].values.reshape(-1,1))**2)

sep=500
max_dist=20001
sv_i=np.zeros(len(range(0,max_dist,sep)))
for i,value in enumerate(tqdm(range(0,max_dist,sep))):
    mask1=xy_dis>value
    mask2=xy_dis<value+sep
    mask=mask1*mask2
    res1=t_vario[mask]
    mask3=res1>0
    res2=(res1[mask3].mean())/2
    sv_i[i]=res2

x=range(0,max_dist,sep)
plt.plot(x[:],sv_i[:],c='black',marker='o')
plt.plot(x[:],vm.spherical_variogram_model([2200-200,4000,200],np.array(x))[:],c='red')

# %%

# %%
parameters = {
    "depth0": [{'sill': 2500-500, 'range': 6000, 'nugget': 500}, 'spherical'],
    "depth500": [{'sill': 3000-500, 'range': 5000, 'nugget': 500}, "spherical"],
    "depth800": [{'sill': 2000-250, 'range': 3000, 'nugget': 250}, 'spherical'],
    "depth1000": [{'sill': 3000-1250, 'range': 20000, 'nugget': 1250}, 'gaussian'],
    # "depth1000": [{'sill': 3000-500, 'range': 4000, 'nugget': 500}, 'gaussian'],#'sill': 3000-1250, 'range': 20000, 'nugget': 1250
}
depth_features = ['x', 'y', 'h', 'volcano', 'curie',
       'age_a', 'age_b', 'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4',
       'HCO3', 'anion', 'group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩',
       'group_ja_変成岩', 'group_ja_火成岩', 'age']
#%%
fix_seed(0)
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)
input_data_plane = input_data.groupby(
    ['x', 'y'], as_index=False).mean()

est_data = pd.read_csv('./input_japan/useful_data/est_grid_detail_ja.csv')
est_data = preprocess_grid(est_data)
est_data_plane=est_data.groupby(
    ['x', 'y'], as_index=False).mean()
#%%
train_data = input_data.copy()
train_data.h_z=train_data.h_z.round(-3)
train_data=train_data.groupby(["x","y","h_z"],as_index=False).mean()
#%%
# for depth_target in [0,500,800]:
    # X_train=train_data[train_data.h_z==-depth_target][depth_features+["t"]].reset_index(drop=True)
    # X_input=input_data_plane[depth_features]
    # X_est = est_data_plane[depth_features].astype(float)
    # # stacking_model=Stacking_train_model_ok(X_train,"t",parameters[f"depth{depth_target}"])
    # # input_data_plane[f"depth{depth_target}"] = Stacking_est_ok(stacking_model,X_train,"t",X_input,parameters[f"depth{depth_target}"])
    # # est_data_plane[f"depth{depth_target}"] = Stacking_est_ok(stacking_model,X_train,"t",X_est,parameters[f"depth{depth_target}"])
    # input_data_plane[f"depth{depth_target}"] = Stacking_model_ok(X_train,"t",X_input,parameters[f"depth{depth_target}"])
    # est_data_plane[f"depth{depth_target}"] = Stacking_model_ok(X_train,"t",X_est,parameters[f"depth{depth_target}"])
    
for depth_target in [1000]:
    target="t"
    trn_x=train_data[train_data.h_z==-depth_target][depth_features].reset_index(drop=True)
    trn_y=train_data[train_data.h_z==-depth_target][target].reset_index(drop=True)
    # X_input=input_data_plane[depth_features]
    X_est = est_data_plane[depth_features].astype(float)
    # stacking_model=Stacking_train_model_ok(X_train,"t",parameters[f"depth{depth_target}"])
    # input_data_plane[f"depth{depth_target}"] = Stacking_est_ok(stacking_model,X_train,"t",X_input,parameters[f"depth{depth_target}"])
    # est_data_plane[f"depth{depth_target}"] = Stacking_est_ok(stacking_model,X_train,"t",X_est,parameters[f"depth{depth_target}"])
    ok1 = OrdinaryKriging(trn_x['x'], trn_x['y'], trn_y, variogram_model=parameters[f"depth{depth_target}"]
                             [1], variogram_parameters=parameters[f"depth{depth_target}"][0])
    # ok2 = OrdinaryKriging(trn_x['x'], trn_x['y'], trn_y)
    est_data_plane[f"depth{depth_target}"]  = ok1.execute('points', X_est['x'], X_est['y'])[0].data
est_data_plane=est_data_plane[["x","y","depth1000"]]
est_data_plane.to_csv("./input_japan/depth/database/ordinary_kriging_depth1000_ja.csv",index=False)

# %%
