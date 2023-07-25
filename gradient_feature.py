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
# %%
input_data = pd.read_csv('./input/useful_data/input_data.csv')
input_data
# %%
def preprocess_input(df):
    df['t'] = np.where(df['t'].values <= 0, 0.1, df['t'].values)
    add_input_volcano = pd.read_csv('./input/volcano/add_input_volcano.csv')
    add_input_curie = pd.read_csv('./input/curie_point/add_input_curie.csv')
    add_input_tishitsu = pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
    add_input_onsen = pd.read_csv('./input/onsen/add_input_onsen.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])

    return df


def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('./input/volcano/add_grid_volcano.csv')
    add_grid_curie = pd.read_csv('./input/curie_point/add_grid_curie.csv')
    add_grid_onsen = pd.read_csv('./input/onsen/add_grid_onsen.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])

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

def feature_engineer(df):
    df['age']=(df['age_a']+df['age_b'])/2
    return df
# %%
input_data=preprocess_input(input_data)
input_data = grad_calc(input_data)
input_data = grad_maxmin_calc(input_data)

input_data.columns
# %%
#%%
xy_input_data = input_data.groupby(['x', 'y'], as_index=False).mean()
xy_input_data = xy_input_data[['x', 'y', 'h','volcano', 'curie', 'age_a', 'age_b','symbol_freq', 'group_freq', 'lithology_freq', 'Temp', 'pH', 'Na', 'K','Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion', 'grad', 'grad_max','grad_min']]
xy_input_data=feature_engineer(xy_input_data)
xy_input_data.describe()
#%%
features = ['x', 'y', 'h', 'volcano', 'curie', 'age_a', 'age_b', 'symbol_freq',
       'group_freq', 'lithology_freq', 'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg',
       'Cl', 'SO4', 'HCO3', 'anion', 'age']
target = 'grad'
trn=xy_input_data[features]
tst=xy_input_data[target]
# trn_int=trn[trn.columns[trn.dtypes==np.int64]]
# trn_float=trn[trn.columns[trn.dtypes==np.float64]]
# sc = StandardScaler()
# trn_float=pd.DataFrame(sc.fit_transform(trn_float),columns=trn_float.columns)
# trn=pd.concat([trn_float,trn_int],axis=1)
trn
# %%
lgb_params = {'max_depth': 11,
                      'n_estimators': 79,
                      'learning_rate': 0.04614912281574665,
                      'num_leaves': 2139,
                      'min_child_samples': 15,
                      'random_state': 0}
lgb = LGBMRegressor(**lgb_params)
cv = KFold(n_splits=5, shuffle=True, random_state=0)
result = model_selection.cross_validate(
    lgb, trn, tst, cv=cv, scoring='neg_mean_squared_error')
print(np.sqrt(-result['test_score'].mean()))

#%%
MLA = {'xgb': XGBRegressor(), 'lgbm': LGBMRegressor(), 'lr': LinearRegression(
    ),  'dt': DecisionTreeRegressor(), 'rf': RandomForestRegressor(), 'kn': KNeighborsRegressor()}
for m in MLA.keys():
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    result = model_selection.cross_validate(
        MLA[m], trn, tst, cv=cv, scoring='neg_mean_squared_error')
    print(m, np.sqrt(-result['test_score'].mean()))
#%%
fig, ax = plt.subplots()
ax.grid()

plt.rcParams["font.size"] = 18
plt.rcParams['font.family'] = 'MS Gothic'
ax.set_title(u'地温勾配')
ax.set_xlabel('Algorithm')
ax.set_ylabel('RMSE(℃/km)')  
# bar=ax.bar(['XGB','LGBM','LR','DT','RF','KN','OK'],[58.42343505069611,56.14227491470641,60.49322774500824,73.07238466248958,56.14084516723853,60.67745076841629,62.41538730670552])
bar=ax.bar(['DT','OK','KN','LR','XGB','LGBM','RF','Stack'],[73.07238466248958,62.41538730670552,60.67745076841629,60.49322774500824,58.42343505069611,56.14227491470641,56.14084516723853,54.79215225405282])

# bar[4].set_color("red")
# bar[5].set_color("red")
bar[7].set_color("red")

plt.show()

#%%
vote_est = [
    ('rf0', RandomForestRegressor(random_state=0)),
    ('rf1', RandomForestRegressor(random_state=1)),
    ('rf2', RandomForestRegressor(random_state=2)),
    ('lgbm0', LGBMRegressor(random_state=0)),
    ('lgbm1', LGBMRegressor(random_state=1)),
    ]
vr=VotingRegressor(estimators=vote_est)
cv = KFold(n_splits=5, shuffle=True, random_state=0)
result = model_selection.cross_validate(
    vr, trn, tst, cv=cv, scoring='neg_mean_squared_error')
print('vr', np.sqrt(-result['test_score'].mean()))
# %%
rmse_list=[]
for trn_idx, val_idx in cv.split(trn, tst):
    trn_x = trn.iloc[trn_idx, :].reset_index(drop=True)
    trn_y = tst[trn_idx].reset_index(drop=True)
    val_x = trn.iloc[val_idx, :].reset_index(drop=True)
    val_y = tst[val_idx].reset_index(drop=True)

    test_df=pd.DataFrame(np.zeros((val_x.shape[0],3)))

    rf=RandomForestRegressor()
    lgbm=LGBMRegressor()
    xgb=XGBRegressor()

    rf.fit(trn_x, trn_y)
    lgbm.fit(trn_x, trn_y)
    xgb.fit(trn_x, trn_y)

    test_df[0]=rf.predict(val_x)
    test_df[1]=lgbm.predict(val_x)
    test_df[2]=xgb.predict(val_x)

    train_df=pd.DataFrame(np.zeros((trn_x.shape[0],3)))
    for trn_idxidx, val_idxidx in cv.split(trn_x, trn_y):
        trn_xx = trn_x.iloc[trn_idxidx, :]
        trn_yy = trn_y[trn_idxidx]
        val_xx = trn_x.iloc[val_idxidx, :]
        val_yy = trn_y[val_idxidx]

        rf=RandomForestRegressor()
        lgbm=LGBMRegressor()
        xgb=XGBRegressor()

        rf.fit(trn_xx, trn_yy)
        lgbm.fit(trn_xx, trn_yy)
        xgb.fit(trn_xx, trn_yy)

        train_df.iloc[val_idxidx, 0]=rf.predict(val_xx)
        train_df.iloc[val_idxidx, 1]=lgbm.predict(val_xx)
        train_df.iloc[val_idxidx, 2]=xgb.predict(val_xx)

    lr=LinearRegression()
    lr.fit(train_df,trn_y)
    pred=lr.predict(test_df)
    rmse_list.append(np.sqrt(mean_squared_error(val_y.values,pred)))
print(np.mean(rmse_list))
#%%

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
#%%
rmse_list=[]
trn=xy_input_data[features+[target]]
for trn_idx, val_idx in cv.split(trn):
    trn_x = trn.iloc[trn_idx, :]
    val_x = trn.iloc[val_idx, :]
    model_s=Stacking_train_model(trn_x,target)
    pred_s=Stacking_est(model_s,trn_x,target,val_x.drop(target,axis=1))
    rmse_list.append(np.sqrt(mean_squared_error(val_x[target].values,pred_s)))
print(np.mean(rmse_list))
#%%
xy_dis=squareform(pdist(trn[['x','y']]))
t_vario=squareform(pdist(tst.values.reshape(-1,1))**2)
sep=110000
max_dist=1500001
sv_i=np.zeros(len(range(0,max_dist,sep)))
for i,value in enumerate(tqdm(range(0,max_dist,sep))):
    mask1=xy_dis>value
    mask2=xy_dis<value+sep
    mask=mask1*mask2
    res1=t_vario[mask]
    mask3=res1>0
    res2=(res1[mask3].mean())/2
    sv_i[i]=res2
sv_i

x=range(0,max_dist,sep)
plt.plot(x[:],sv_i[:],c='black',marker='o')
plt.plot(x[:],vm.spherical_variogram_model([1000,600000,4250],np.array(x))[:],c='red')

#%%
rmse_list=[]
for trn_idx, val_idx in cv.split(trn, tst):
    trn_x = trn.iloc[trn_idx, :]
    trn_y = tst[trn_idx]
    
    val_x = trn.iloc[val_idx, :]
    val_y = tst[val_idx]

    ok = OrdinaryKriging(trn_x['x'], trn_x['y'], trn_y,variogram_model='spherical')#,variogram_parameters={'sill':1000,'range':600000,'nugget':4250}) 
    predict=ok.execute('points',val_x['x'], val_x['y'])[0].data
    rmse_list.append(np.sqrt(mean_squared_error(val_y.values,predict)))
print(np.mean(rmse_list))
#%%
x, y = xy_input_data[xy_input_data['grad_min'] ==
                     xy_input_data['grad_min'].min()][['x', 'y']].values[0]
a = input_data[(input_data['x'] == x) & (input_data['y'] == y)]
plt.plot(a['t'], -a['z'])
print(x, y)
# %%
# input_data_master.to_csv('./input/gradient/add_input_grad.csv',index=False)
# %%
# grid推定(ok)
# %%
input_ok = grad_data_master.values
input_ok
grad_data_master
# %%
xy_dis = squareform(pdist(grad_data_master[['x', 'y']]))
# %%
t_vario = squareform(pdist(grad_data_master['grad'].values.reshape(-1, 1))**2)
# %%
sep = 110
max_dist = 20001
sv_i = np.zeros(len(range(0, max_dist, sep)))
for i, value in enumerate(tqdm(range(0, max_dist, sep))):
    mask1 = xy_dis > value
    mask2 = xy_dis < value+sep
    mask = mask1*mask2
    res1 = t_vario[mask]
    mask3 = res1 > 0
    res2 = (res1[mask3].mean())/2
    sv_i[i] = res2
sv_i
# %%
x = range(0, max_dist, sep)
plt.plot(x[:], sv_i[:], c='black', marker='o')
plt.plot(x[:], vm.spherical_variogram_model(
    [0.01, 200000, 0.1], np.array(x))[:], c='red')

# %%
okd = OrdinaryKriging(input_ok[:, 0], input_ok[:, 1], input_ok[:, 2],
                      variogram_model='spherical')
# %%
est_data_master = pd.read_csv('./input/useful_data/est_grid_500.csv')
est_data_xy = est_data_master.groupby(
    ['x', 'y'], as_index=False).mean()[['x', 'y']]
est_data = est_data_xy.values.astype('float')
est_data.shape
# %%
predict = okd.execute('points', est_data[:, 0], est_data[:, 1])[0].data
predict
# %%
est_data_xy['grad'] = predict
est_data_xy
# %%
# est_data_xy.to_csv('./input/gradient/add_grid_grad.csv',index=False)
# %%
