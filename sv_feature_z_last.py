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
#%%
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

# def torch_select_column_by_condition_2(data, index, condition):
#     condition_data = data[:, index]
#     mask = condition_data.eq(condition)
#     mask, mask_ = mask.min(1)
#     if len(torch.nonzero(mask)) == 0:
#         return torch.Tensor()
#     indices = torch.squeeze(torch.nonzero(mask), 1)
#     select = torch.index_select(data, 0, indices)
#     return select

def torch_select_column_by_condition_2(data, index, condition):
    condition_data = data[:, index]
    mask1 = condition_data[:,index[0]].eq(condition[0])
    mask2 = condition_data[:,index[1]].eq(condition[1])
    mask=mask1*mask2
    if len(torch.nonzero(mask)) == 0:
        return torch.Tensor()
    indices = torch.squeeze(torch.nonzero(mask), 1)
    select = torch.index_select(data, 0, indices)
    return select
#%%

def calc_model_semivariogram_z(df_input):
    df_input = df_input.copy()
    # df_input.h_z = df_input.h_z.round(-2)
    df_input_xy=df_input.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    sv_all=[]
    for x,y in df_input_xy.values:
        pc=df_input[(df_input.x==x)&(df_input.y==y)]
        h_z=pc[["h_z"]].values
        t=pc[["t"]].values
        dist_z=squareform(pdist(h_z))
        dist_t=squareform(pdist(t))**2
        sv_i=np.zeros(len(range(250,1001,250)))#1501
        for i,dist in enumerate(range(250,1001,250)):#1501
            # mask=dist_z==dist
            mask1 = dist_z>=dist-50
            mask2 = dist_z<=dist+50
            res=dist_t[mask1*mask2]
            if res.size:
                sv_i[i]=res.mean()
            else:
                sv_i[i]=np.nan
        sv_all.append(sv_i/2)
    sv_all=np.vstack(sv_all)
    sv_all=pd.DataFrame(sv_all,columns=["sv_250","sv_500","sv_750","sv_1000"])
    df_input_xy = pd.concat([df_input_xy,sv_all],axis=1)
    return df_input_xy

def create_d_z_list(ts_est,ts_est_xy):
    dist_z_list=[]
    for xy in ts_est_xy:
        pc=torch_select_column_by_condition_2(ts_est,[0,1],xy)
        h_z=pc[:,[2]]
        dist_z=torch.sqrt(pairwise_distances(h_z))
        dist_z_list.append(dist_z)
    return dist_z_list

def calc_pred_semivariogram_z(ts_est,ts_est_xy,dist_z_list):
    sv_all=[]
    for i,xy in enumerate(ts_est_xy):
        pc=torch_select_column_by_condition_2(ts_est,[0,1],xy)
        pred_t=pc[:,[3]]
        dist_z=dist_z_list[i]
        dist_t=pairwise_distances(pred_t)
        sv_i=[]
        for z in range(500,1001,500):#1501
            sv_i.append(dist_t[dist_z==z].mean())
        sv_i=torch.stack(sv_i)
        sv_all.append(sv_i)
    sv_all=torch.stack(sv_all)
    return sv_all


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
#%%
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)
est_data = pd.read_csv('./input_japan/useful_data/est_grid_500_ja.csv')
curie_data = pd.read_csv('./input_japan/curie_point/grid_curie_ja.csv')
curie_data_580ika = pd.read_csv('./input_japan/curie_point/grid_curie_580ika_ja.csv')
curie_data_580izyou = pd.read_csv('./input_japan/curie_point/grid_curie_580izyou_ja.csv')

est_data = preprocess_grid(est_data)
curie_data = preprocess_grid(curie_data)
curie_data_580ika = preprocess_grid(curie_data_580ika)
curie_data_580izyou = preprocess_grid(curie_data_580izyou)
            

#depth feature
# if "depth" in name:
add_input_depth = pd.read_csv(f'./input_japan/depth/add_input_depth_ja.csv')
add_grid_depth = pd.read_csv(f'./input_japan/depth/add_grid_depth_detail_ja.csv')

input_data = input_data.merge(add_input_depth, how='left', on=['x', 'y'])
# test_data = test_data.merge(add_input_depth, how='left', on=['x', 'y'])
est_data = est_data.merge(add_grid_depth, how='left', on=['x', 'y'])
curie_data=curie_data.merge(add_grid_depth, how='left', on=['x', 'y'])
curie_data_580ika=curie_data_580ika.merge(add_grid_depth, how='left', on=['x', 'y'])
curie_data_580izyou=curie_data_580izyou.merge(add_grid_depth, how='left', on=['x', 'y'])

# grad feature
# if "grad" in name:
# add_input_grad = pd.read_csv(f'./input_japan/grad/add_input_grad_ja_inter_{num}.csv')
input_data = grad_calc(input_data)
input_data = grad_maxmin_calc(input_data)
add_grid_grad = pd.read_csv(f'./input_japan/grad/add_grid_grad_detail_ja.csv')

# input_data = input_data.merge(add_input_grad, how='left', on=['x', 'y'])
# test_data = test_data.merge(add_input_grad, how='left', on=['x', 'y'])
est_data = est_data.merge(add_grid_grad, how='left', on=['x', 'y'])
curie_data=curie_data.merge(add_grid_grad, how='left', on=['x', 'y'])
curie_data_580ika=curie_data_580ika.merge(add_grid_grad, how='left', on=['x', 'y'])
curie_data_580izyou=curie_data_580izyou.merge(add_grid_grad, how='left', on=['x', 'y'])
#%%
# %%
input_sv_plane=calc_model_semivariogram_z(input_data)
input_sv_plane
# %%
features =['x', 'y', 'h', 'volcano', 'curie',
       'age_a', 'age_b', 'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4',
       'HCO3', 'anion', 'group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩',
       'group_ja_変成岩', 'group_ja_火成岩', 'age',"depth0","depth500","depth1000",'grad','grad_min','grad_max_h_z','grad_min_h_z']
# %%
xy_input_data = input_data.groupby(['x', 'y'], as_index=False).mean()
#%%
result_df=[]
for key in ["sv_250","sv_500","sv_750","sv_1000"]:
    result_dict={}
    target = key
    sv_input_data = xy_input_data.merge(input_sv_plane,on=["x","y"],how="left")
    sv_input_data = sv_input_data[~sv_input_data[key].isnull()]
    
    trn=sv_input_data[features]
    tst=sv_input_data[target]
    print(f'{key}', '-'*10, tst.std())
    
    MLA = {'xgb': XGBRegressor(random_state=0), 'lgbm': LGBMRegressor(random_state=0), 'rf': RandomForestRegressor(random_state=0)}
    for m in MLA.keys():
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        result = model_selection.cross_validate(
            MLA[m], trn, tst, cv=cv, scoring='neg_mean_squared_error')
        print(m, np.sqrt(-result['test_score'].mean()))
        result_dict[m]=np.sqrt(-result['test_score'].mean())
        
    start_time = time.time()
    rmse_list=[]
    trn=sv_input_data[features]
    tst=sv_input_data[target]
    trn=pd.concat([trn,tst],axis=1)
    for trn_idx, val_idx in cv.split(trn):
        trn_x = trn.iloc[trn_idx, :]
        val_x = trn.iloc[val_idx, :]
        # model_s=Stacking_train_model(trn_x,key)
        # pred_s=Stacking_est(model_s,trn_x,key,val_x.drop(key,axis=1))
        pred_s=Stacking_model(trn_x,key,val_x.drop(key,axis=1))
        rmse_list.append(np.sqrt(mean_squared_error(val_x[key].values,pred_s)))
    print("st",np.mean(rmse_list),time.time()-start_time)
    result_dict['st']=np.mean(rmse_list)
    
    result_dict=pd.DataFrame(result_dict.values(),index=result_dict.keys(),columns=[key])
    # print(result_dict)
    result_df.append(result_dict)
result_df=pd.concat(result_df,axis=1)
#%%
result_df.to_csv("./input_japan/sv/pred_sv_result.csv")
#%%
# all
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)
input_data = grad_calc(input_data)
input_data = grad_maxmin_calc(input_data)
input_data_plane = input_data.groupby(
    ['x', 'y'], as_index=False).mean()
add_input_depth = pd.read_csv("./input_japan/depth/add_input_depth_ja.csv")
add_input_sv =calc_model_semivariogram_z(input_data)
input_data_plane=input_data_plane.merge(add_input_depth,on=["x","y"],how="left")
input_data_plane=input_data_plane.merge(add_input_sv,on=["x","y"],how="left")

est_data = pd.read_csv('./input_japan/useful_data/est_grid_detail_ja.csv')
est_data = preprocess_grid(est_data)
est_data_plane=est_data.groupby(
    ['x', 'y'], as_index=False).mean()
add_grid_depth = pd.read_csv("./input_japan/depth/add_grid_depth_detail_ja.csv")
add_grid_grad = pd.read_csv("./input_japan/grad/add_grid_grad_detail_ja.csv")

est_data_plane=est_data_plane.merge(add_grid_depth,on=["x","y"],how="left")
est_data_plane=est_data_plane.merge(add_grid_grad,on=["x","y"],how="left")

for sv_target in ['sv_250','sv_1000']:
    X_train = input_data_plane[features+[sv_target]].dropna().reset_index(drop=True)
    X_est = est_data_plane[features]
    # stacking_model=Stacking_train_model(X_train,sv_target)
    # est_data_plane[sv_target] = Stacking_est(stacking_model,X_train,sv_target,X_est)
    est_data_plane[sv_target] = Stacking_model(X_train,sv_target,X_est)
for sv_target in['sv_500']:
    X_train_raw = input_data_plane[features+[sv_target]].dropna().reset_index(drop=True)
    X_train = X_train_raw[features]
    Y_train = X_train_raw[sv_target]
    X_est = est_data_plane[features]
    rf = XGBRegressor(random_state=0)
    rf.fit(X_train,Y_train)
    est_data_plane[sv_target] = rf.predict(X_est)

for sv_target in['sv_750']:
    X_train_raw = input_data_plane[features+[sv_target]].dropna().reset_index(drop=True)
    X_train = X_train_raw[features]
    Y_train = X_train_raw[sv_target]
    X_est = est_data_plane[features]
    rf = RandomForestRegressor(random_state=0)
    rf.fit(X_train,Y_train)
    est_data_plane[sv_target] = rf.predict(X_est)
est_data_plane=est_data_plane[["x","y"]+["sv_250","sv_500","sv_750","sv_1000"]]
est_data_plane.to_csv("./input_japan/sv/add_grid_sv_detail.csv",index=False)
#%%
add_input_sv.describe()
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
variogram(input_sv_plane[["x","y","sv_750"]].dropna().reset_index(drop=True), sep=110, max_dist=2001, parameters={
        'sill': 5000-4000, 'range': 10000, 'nugget': 4000}, how='spherical')
#%%

def calc_model_semivariogram_z(df_input):
    df_input = df_input.copy()
    # df_input.h_z = df_input.h_z.round(-2)
    df_input_xy=df_input.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    sv_all=[]
    for x,y in df_input_xy.values:
        pc=df_input[(df_input.x==x)&(df_input.y==y)]
        h_z=pc[["h_z"]].values
        t=pc[["t"]].values
        dist_z=squareform(pdist(h_z))
        dist_t=squareform(pdist(t))**2
        sv_i=np.zeros(len(range(0,5001,100)))#1501
        for i,dist in enumerate(range(0,5001,100)):#1501
            # mask=dist_z==dist
            mask1 = dist_z>=dist-50
            mask2 = dist_z<=dist+50
            res=dist_t[mask1*mask2]
            if res.size:
                sv_i[i]=res.mean()
            else:
                sv_i[i]=np.nan
        sv_all.append(sv_i)
    sv_all=np.vstack(sv_all)
    # sv_all=pd.DataFrame(sv_all,columns=["sv_250","sv_500","sv_750","sv_1000"])
    # df_input_xy = pd.concat([df_input_xy,sv_all],axis=1)
    return sv_all
# %%
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')

# %%
sv_all = calc_model_semivariogram_z(input_data)
# %%
import seaborn as sns
fig,ax=plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
ax.set_xlabel('Distance between data(m)') 
ax.set_ylabel('Semivariogram')
ax.plot(np.arange(0,5001,125)[::2],np.nanmean(sv_all,axis=0)[::2]/2,c='black', marker='o')
ax.plot( np.arange(0,5001,125), vm.gaussian_variogram_model(
            [7500, 1800, 0], np.arange(0,5001,125)), c='red')
# %%
sv_all.shape

# %%
a = input_data.groupby(["x","y"],as_index=False).mean()

# %%
import seaborn as sns
fig,ax=plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
ax.set_xlabel('Distance between data(m)') 
ax.set_ylabel('Semivariogram')

# ax.set_title("内挿評価")
# ax.set_xlim(0,800)
ax.plot(np.arange(0,5001,100),sv_all[np.argmax(a.t.values),:]/2,c="red")
ax.plot(np.arange(0,5001,100),sv_all[np.argmin(a.t.values)-1,:]/2,c="blue")

ax.legend(["地熱地域","非地熱地域"])
# fig.savefig(f"./修論用画像/セミバリオグラム例.png",dpi=300,bbox_inches='tight')
# %%
b=input_data[(input_data.x==a.iloc[np.argmin(a.t.values)-1,:].x)&(input_data.y==a.iloc[np.argmin(a.t.values)-1,:].y)]
plt.plot(b.t,b.h_z)
# %%
print(a.iloc[np.argmax(a.t.values),:].x,a.iloc[np.argmax(a.t.values),:].y)
print(a.iloc[np.argmin(a.t.values)-1,:].x,a.iloc[np.argmin(a.t.values)-1,:].y)
# %%
