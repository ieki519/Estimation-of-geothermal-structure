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
import folium
#%%
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpys
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
        folium.CircleMarker(location=[df["ido"][i],df["keido"][i]],radius=0.1).add_to(m)
        
    return m
#%%
master_features=['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
                    'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
                        'group_rank', 'age']

# preprocess
input_data = pd.read_csv('./input/useful_data/input_data.csv')
input_data = preprocess_input(input_data)


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

train_data_plane = input_data.groupby(['x', 'y'], as_index=False).mean()
est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

for depth_target in depth_target_list:
    train_data_depth = input_data[input_data.h_z==-depth_target].copy()
    
    depth_target=f"depth{depth_target}"
    train_data_depth[depth_target] =train_data_depth["t"]
    
    X_depth = train_data_depth[depth_features+[depth_target]]
    X_train = train_data_plane[depth_features]
    X_est = est_data_plane[depth_features]

    stacking_model=Stacking_train_model(X_depth,depth_target)
    train_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_train)
    est_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_est)
    
    input_data = input_data.merge(train_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
    est_data = est_data.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
    curie_data=curie_data.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
    curie_data_580ika=curie_data_580ika.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
    curie_data_580izyou=curie_data_580izyou.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
    master_features += [depth_target]

# grad feature
input_data = grad_calc(input_data)
input_data=grad_maxmin_calc(input_data)

train_data_plane = input_data.groupby(
    ['x', 'y'], as_index=False).mean()

est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

grad_features = master_features.copy()
grad_target_list = ['grad','grad_max','grad_min']

for grad_target in grad_target_list:
    X_train = train_data_plane[grad_features+[grad_target]]
    X_est = est_data_plane[grad_features]

    stacking_model=Stacking_train_model(X_train,grad_target)
    est_data_plane[grad_target] = Stacking_est(stacking_model,X_train,grad_target,X_est)
    
    est_data = est_data.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
    curie_data=curie_data.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
    curie_data_580ika=curie_data_580ika.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
    curie_data_580izyou=curie_data_580izyou.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
    master_features += [grad_target]
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
    return sv_all
#%%
sv_all = calc_model_semivariogram_z(input_data,est_data)
sv_all=pd.DataFrame(sv_all,columns=["sv_500","sv_1000","sv_1500"])
sv_all
# %%
sv_all.isnull().sum()
#%%
input_data_plane = input_data.groupby(["x","y"],as_index=False).mean()
input_data_plane
# %%
input_data_sv = pd.concat([input_data_plane,sv_all],axis=1)
input_data_sv
# %%
input_data_sv.columns
#%%
sv_features =['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
       'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
        'group_rank', 'age', 'depth0', 'depth500', 'depth800', 'grad',
       'grad_max', 'grad_min',"sv_500"]
#%%
input_data_sv= input_data_sv[sv_features]
input_data_sv = input_data_sv.dropna().reset_index(drop=True)
input_data_sv
#%%
df=input_data_sv.copy()

result_dict={}
trn = df.drop(columns="sv_500")
tst = df["sv_500"]

MLA = {'xgb': XGBRegressor(random_state=0), 'lgbm': LGBMRegressor(random_state=0), 'lr': LinearRegression(
),  'dt': DecisionTreeRegressor(), 'rf': RandomForestRegressor(random_state=0), 'kn': KNeighborsRegressor()}
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
                            , variogram_parameters=[2800-1750,4000,1750])
    ok2 = OrdinaryKriging(trn_x['x'], trn_x['y'], trn_y)
    predict1 = ok1.execute('points', val_x['x'], val_x['y'])[0].data
    predict2 = ok2.execute('points', val_x['x'], val_x['y'])[0].data
    rmse_list1.append(np.sqrt(mean_squared_error(val_y.values, predict1)))
    rmse_list2.append(np.sqrt(mean_squared_error(val_y.values, predict2)))

print('ok1', np.mean(rmse_list1))
print('ok2', np.mean(rmse_list2))
result_dict['ok1']=np.mean(rmse_list1)
result_dict['ok2']=np.mean(rmse_list2)
#%%
# Voting
vote_est = [
    ('rf0', RandomForestRegressor(random_state=0)),
    # ('lgbm0', LGBMRegressor(random_state=0)),
    ('xgb0', XGBRegressor(random_state=0)),
    ('xgb1', XGBRegressor(random_state=1)),
    # ('xgb2', XGBRegressor(random_state=2)),
    ]
vr=VotingRegressor(estimators=vote_est)
cv = KFold(n_splits=5, shuffle=True, random_state=0)
result = model_selection.cross_validate(
    vr, trn, tst, cv=cv, scoring='neg_mean_squared_error')
print('vr', np.sqrt(-result['test_score'].mean()))
result_dict['vr']=np.sqrt(-result['test_score'].mean())
#%%
# Stacking
rmse_list=[]
trn["sv_500"]=tst
for trn_idx, val_idx in cv.split(trn):
    trn_x = trn.iloc[trn_idx, :]
    val_x = trn.iloc[val_idx, :]
    model_s=Stacking_train_model(trn_x,"sv_500")
    pred_s=Stacking_est(model_s,trn_x,"sv_500",val_x.drop("sv_500",axis=1))
    rmse_list.append(np.sqrt(mean_squared_error(val_x["sv_500"].values,pred_s)))
print("st",np.mean(rmse_list))
result_dict['stacking']=np.mean(rmse_list)

result_dict=pd.DataFrame(result_dict.values(),index=result_dict.keys(),columns=["sv_500"])
#%%


#%%
xy_dis=squareform(pdist(input_data_sv[['x','y']]))
t_vario=squareform(pdist(input_data_sv['t'].values.reshape(-1,1))**2)

sep=1100
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
plt.plot(x[:],vm.spherical_variogram_model([2800-1750,4000,1750],np.array(x))[:],c='red')

#%%


#%%
# wgs_to_albers = pd.read_csv("./input/WGS_to_albers/input_WGS_albers.csv")
# wgs_to_albers.x ,wgs_to_albers.y = wgs_to_albers.x.round(),wgs_to_albers.y.round()
# wgs_to_albers=wgs_to_albers.groupby(["x","y"],as_index=False).mean()
# wgs_to_albers
# # %%
# input_data_sv=input_data_sv.merge(wgs_to_albers,on=["x","y"],how="left")
# input_data_sv
# # %%
# df = input_data_sv[["ido","keido","sv_1500"]]
# df = df.dropna().reset_index(drop=True)
# df
# # %%
# visualize_locations(df)
# %%
