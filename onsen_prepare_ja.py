# %%
from os import replace
from pykrige.ok import OrdinaryKriging
from threading import local
import pandas as pd
import numpy as np
import codecs
from scipy.spatial.distance import pdist, squareform
from seaborn.external.docscrape import Parameter
from tqdm import tqdm
import matplotlib.pyplot as plt
import pykrige.variogram_models as vm
import seaborn as sns
import time
import json
import requests
from sklearn.model_selection import train_test_split
import time
import xgboost as xgb
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
import random
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
fix_seed(0)

# %%
# onsen = pd.read_csv("onsen_xyh.csv")
# onsen = onsen.drop(columns=["x","y"])
# onsen.ido,onsen.keido = onsen.ido.round(8),onsen.keido.round(8)
# onsen
# #%%
# WGS_albers_ja = pd.read_csv("onsen_WGS_albers_ja.csv")
# WGS_albers_ja.ido,WGS_albers_ja.keido = WGS_albers_ja.ido.round(8),WGS_albers_ja.keido.round(8)
# WGS_albers_ja
# #%%
# onsen = onsen.merge(WGS_albers_ja,how="left",on=["ido","keido"])
# onsen.to_csv("onsen_xyh_ja.csv",index=False)
#%%
def encode_top(s, lim):  # Seriesを入力
    uniqs, freqs = np.unique(s, return_counts=True)
    top = sorted(zip(uniqs, freqs), key=lambda x: x[1], reverse=True)[:lim]
    top_map = {uf[0]: lank for uf, lank in zip(top, range(len(top)))}
    return s.map(lambda x: top_map.get(x, lim)).astype(np.int),top_map


def one_hot_encode(df, col):
    tmp = pd.get_dummies(df[col], col)
    df = pd.concat([df, tmp], axis=1)
    df.drop(col, axis=1, inplace=True)
    return df

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

# %%
onsen = pd.read_csv('./input_japan/onsen/database/onsen_xyh_ja.csv')
onsen = onsen[~(onsen['h'] == '-----')]

onsen['x'], onsen['y'] = onsen['x'].round(5), onsen['y'].round(5)
onsen_features = ['x', 'y', 'h',
                  'Simplified_rock_category']#, 'Rock_age_EN', 'Litho_EN'
onsen_features = onsen[onsen_features].copy()
onsen_features['Simplified_rock_category'] = onsen_features['Simplified_rock_category'].fillna(
    'nan')
onsen_features[['x', 'y', 'h']] = onsen_features[[
    'x', 'y', 'h']].astype(np.float)
rock_cat_dict={'sediments':'堆積岩', 'volcanic':'火成岩', 'Quat sediments':'堆積岩', 'nan':'その他',
       'accretionary complex':'付加体', 'mafic plutonic':'火成岩', 'granitic':'火成岩', 'ultramafic':'火成岩',
       'felsic plutonic':'火成岩', 'gneiss and schist':'変成岩'}
onsen_features['Simplified_rock_category']=onsen_features['Simplified_rock_category'].apply(lambda x:rock_cat_dict[x])
onsen_features['Simplified_rock_category'],top_map = encode_top(onsen_features['Simplified_rock_category'], 5)
onsen_features = one_hot_encode(onsen_features, 'Simplified_rock_category')

add_onsen_volcano=pd.read_csv('./input_japan/onsen/add_onsen_volcano_ja.csv')
add_onsen_curie=pd.read_csv('./input_japan/onsen/add_onsen_curie_ja.csv')
add_onsen_volcano['x'], add_onsen_volcano['y'] = add_onsen_volcano['x'].round(5), add_onsen_volcano['y'].round(5)
add_onsen_curie['x'], add_onsen_curie['y'] = add_onsen_curie['x'].round(5), add_onsen_curie['y'].round(5)
onsen_features=onsen_features.merge(add_onsen_volcano,on=['x','y'],how='left')
onsen_features=onsen_features.merge(add_onsen_curie,on=['x','y'],how='left')
onsen_features.columns
# onsen_features[onsen_features['Litho_EN']=='-----']
# onsen_features.astype(float)
# %%
# onsen_features = onsen_features[['x', 'y', 'h']]
# print(onsen_features.columns)
# %%
Temp = onsen.loc[:, ['x', 'y', 'Temp']].dropna().astype(float)
pH = onsen.loc[:, ['x', 'y', 'pH']].dropna().astype(float)
Na = onsen.loc[:, ['x', 'y', 'Na']].dropna().astype(float)
K = onsen.loc[:, ['x', 'y', 'K']].dropna().replace(
    ['<0.1', '<1.0'], 0).astype(float)
Ca = onsen.loc[:, ['x', 'y', 'Ca']].dropna().replace(['<1.0'], 0).astype(float)
Mg = onsen.loc[:, ['x', 'y', 'Mg']].dropna().replace(
    ['<0.1', '<1.0', '<0.01', '<0.02', 'TR'], 0).astype(float)
Cl = onsen.loc[:, ['x', 'y', 'Cl']].dropna().replace(['TR'], 0).astype(float)
SO4 = onsen.loc[:, ['x', 'y', 'SO4']].dropna().replace(
    ['<5', '<0.05', 'N.D.', '<0.1', '<1', '<1.0', 'TR'], 0).astype(float)
HCO3 = onsen.loc[:, ['x', 'y', 'HCO3']].dropna().replace(
    ['TR', 'N.D.', '<0.6', '<10', '<0.01', '<0.1'], 0).astype(float)
# %%
# Temp = Temp
# pH = pH
# Na.Na = Na.Na*(1/22.989769280)
# K.K = K.K*(1/39.09830)
# Ca.Ca = Ca.Ca*(2/40.0780)
# Mg.Mg = Mg.Mg*(2/24.30500)
# Cl.Cl = Cl.Cl
# SO4.SO4 = SO4.SO4
# HCO3.HCO3 = HCO3.HCO3
#%%
Temp = Temp.merge(onsen_features, on=['x', 'y'], how='left').dropna()
pH = pH.merge(onsen_features, on=['x', 'y'], how='left').dropna()
Na = Na.merge(onsen_features, on=['x', 'y'], how='left').dropna()
K = K.merge(onsen_features, on=['x', 'y'], how='left').dropna()
Ca = Ca.merge(onsen_features, on=['x', 'y'], how='left').dropna()
Mg = Mg.merge(onsen_features, on=['x', 'y'], how='left').dropna()
Cl = Cl.merge(onsen_features, on=['x', 'y'], how='left').dropna()
SO4 = SO4.merge(onsen_features, on=['x', 'y'], how='left').dropna()
HCO3 = HCO3.merge(onsen_features, on=['x', 'y'], how='left').dropna()
# %%
anion = onsen.loc[:, ['x', 'y', 'SO4', 'Cl', 'HCO3']].dropna().replace(
    ['<5', '<0.05', 'N.D.', '<0.1', '<1', '<1.0', 'TR', '<0.6', '<10', '<0.01'], 0).astype(float)
anion_SO4 = anion['SO4'].values*(2/96.0626)
anion_Cl = anion['Cl'].values*(1/35.4530)
anion_HCO3 = anion['HCO3'].values*(1/61.0168)
anion_index = 0.5*(anion_SO4/(anion_Cl+anion_SO4) +
                   (anion_Cl+anion_SO4)/(anion_Cl+anion_SO4+anion_HCO3))
anion['anion'] = anion_index
anion = anion.loc[:, ['x', 'y', 'anion']]
anion = anion.merge(onsen_features, on=['x', 'y'], how='left').dropna()
#%%
parameters = {
    'Temp': [{'sill': 300, 'range': 75000, 'nugget': 175}, 'spherical'],
    'pH': [{'sill': 1.75-0.5, 'range': 2500, 'nugget': 0.5}, 'gaussian'],
    'Na': [{'sill': 2250000-800000, 'range': 7500, 'nugget': 800000}, 'spherical'],
    'K': [{'sill': 60000, 'range': 25000, 'nugget': 20000}, 'gaussian'],
    'Ca': [{'sill': 150000-50000, 'range': 150, 'nugget': 50000}, 'spherical'],
    'Mg': [{'sill': 16000, 'range': 180000, 'nugget': 12000}, 'gaussian'],
    'Cl': [{'sill': 9000000-3500000, 'range': 8000, 'nugget': 3500000}, 'spherical'],
    'SO4': [{'sill': 220000, 'range': 2000, 'nugget': 80000}, 'gaussian'],
    'HCO3': [{'sill': 500000, 'range': 50000, 'nugget': 200000}, 'gaussian'],
    'anion': [{'sill': 0.03-0.007, 'range': 5000, 'nugget': 0.007}, 'gaussian']
}
# %%
result_df=[]
onsen_dict = {'Temp': Temp, 'pH': pH, 'Na': Na, 'K': K, 'Ca': Ca,
              'Mg': Mg, 'Cl': Cl, 'SO4': SO4, 'HCO3': HCO3, 'anion': anion}
# onsen_dict={'anion': anion}
for key in onsen_dict.keys():
    result_dict={}
    trn = onsen_dict[key].drop(columns=key)
    tst = onsen_dict[key][key]
    print(f'{key}', '-'*10, tst.std())
    MLA = {'xgb': XGBRegressor(), 'lgbm': LGBMRegressor(), 'rf': RandomForestRegressor()}
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
        trn_y = tst[trn_idx]

        val_x = trn.iloc[val_idx, :]
        val_y = tst[val_idx]

        ok1 = OrdinaryKriging(trn_x['x'], trn_x['y'], trn_y, variogram_model=parameters[key]
                             [1], variogram_parameters=parameters[key][0])
        # ok2 = OrdinaryKriging(trn_x['x'], trn_x['y'], trn_y)
        predict1 = ok1.execute('points', val_x['x'], val_x['y'])[0].data
        # predict2 = ok2.execute('points', val_x['x'], val_x['y'])[0].data
        rmse_list1.append(np.sqrt(mean_squared_error(val_y.values, predict1)))
        # rmse_list2.append(np.sqrt(mean_squared_error(val_y.values, predict2)))

    print('ok1', np.mean(rmse_list1))
    # print('ok2', np.mean(rmse_list2))
    result_dict['ok1']=np.mean(rmse_list1)
    # result_dict['ok2']=np.mean(rmse_list2)
    
    start_time = time.time()
    rmse_list=[]
    trn = onsen_dict[key].drop(columns=key)
    tst = onsen_dict[key][key]
    trn=pd.concat([trn,tst],axis=1)
    for trn_idx, val_idx in cv.split(trn):
        trn_x = trn.iloc[trn_idx, :]
        val_x = trn.iloc[val_idx, :]
        # model_s=Stacking_train_model(trn_x,key)
        # pred_s=Stacking_est(model_s,trn_x,key,val_x.drop(key,axis=1))
        pred_s = Stacking_model(trn_x,key,val_x.drop(key,axis=1))
        rmse_list.append(np.sqrt(mean_squared_error(val_x[key].values,pred_s)))
    print("st",np.mean(rmse_list),time.time()-start_time)
    result_dict['st']=np.mean(rmse_list)
    
    start_time = time.time()
    rmse_list=[]
    trn = onsen_dict[key].drop(columns=key)
    tst = onsen_dict[key][key]
    trn=pd.concat([trn,tst],axis=1)
    for trn_idx, val_idx in cv.split(trn):
        trn_x = trn.iloc[trn_idx, :]
        val_x = trn.iloc[val_idx, :]
        # model_s=Stacking_train_model_ok(trn_x,key,parameters[key])
        # pred_s=Stacking_est_ok(model_s,trn_x,key,val_x.drop(key,axis=1),parameters[key])
        pred_s = Stacking_model_ok(trn_x,key,val_x.drop(key,axis=1),parameters[key])
        rmse_list.append(np.sqrt(mean_squared_error(val_x[key].values,pred_s)))
    print("st_ok",np.mean(rmse_list),time.time()-start_time)
    result_dict['st_ok']=np.mean(rmse_list)
    
    result_dict=pd.DataFrame(result_dict.values(),index=result_dict.keys(),columns=[key])
    print(result_dict)
    
    result_df.append(result_dict)
result_df=pd.concat(result_df,axis=1)
#%%
# result_df.to_csv('./input_japan/onsen/database/onsen_pred_result.csv')
# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# result_df=pd.read_csv('./input_japan/onsen/database/onsen_pred_result.csv').rename(columns={'Unnamed: 0':'Model'})
# s=result_df.iloc[[-1,-2],:].min()
# s.name=8
# result_df=result_df.append(s)
# result_df=result_df.drop(index=[6,7])
# result_df=result_df.reset_index(drop=True)
# sns.set()
# plt.rcParams['font.family'] = 'Yu Gothic' # font familyの設定
# plt.rcParams['font.weight']='bold'
# plt.rcParams['axes.labelweight']='bold'
# plt.rcParams['axes.titleweight']='bold'
for key in ['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3','anion']:
    fig, ax = plt.subplots()
    ax.grid()

    plt.rcParams["font.size"] = 18
    
    ax.set_title(key)
    ax.set_xlabel('Algorithm')
    if key=='Temp':
        ax.set_ylabel('RMSE(℃)')  
    elif key=='pH' or key=='anion':
        ax.set_ylabel('RMSE')
    else:
        ax.set_ylabel('RMSE(mg/kg)')
    bar=ax.bar(['XGB','LGBM','RF','OK'],result_df[key])
    bar[result_df[key].idxmin()].set_color("red")
    plt.show()
#%%
def preprocess_input(df):
    add_input_volcano = pd.read_csv('./input_japan/volcano/add_input_volcano_ja.csv')
    add_input_curie = pd.read_csv('./input_japan/curie_point/add_input_curie_ja.csv')
    add_input_tishitsu = pd.read_csv('./input_japan/tishitsu/add_input_tishitsu_pred_ja.csv')
    
    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])

    return df


def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('./input_japan/volcano/add_grid_volcano_detail_ja.csv')
    add_grid_curie = pd.read_csv('./input_japan/curie_point/add_grid_curie_detail_ja.csv')
    add_grid_tishitsu = pd.read_csv('./input_japan/tishitsu/add_grid_tishitsu_detail_pred_ja.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])

    return df

input_data=pd.read_csv('./input_japan/useful_data/input_data_ja.csv').groupby(['x','y'],as_index=False).mean()[['x','y','h']]
grid_data=pd.read_csv('./input_japan/useful_data/est_grid_detail_ja.csv').groupby(['x','y'],as_index=False).mean()[['x','y','h']]
input_data=preprocess_input(input_data)
grid_data=preprocess_grid(grid_data)
input_data['Simplified_rock_category']=input_data['group_ja'].map(lambda x:top_map[x])
grid_data['Simplified_rock_category']=grid_data['group_ja'].map(lambda x:top_map[x])
input_data = one_hot_encode(input_data, 'Simplified_rock_category')
grid_data = one_hot_encode(grid_data, 'Simplified_rock_category')
features=['x', 'y', 'h', 'Simplified_rock_category_0',
       'Simplified_rock_category_1', 'Simplified_rock_category_2',
       'Simplified_rock_category_3', 'Simplified_rock_category_4', 'volcano',
       'curie']
input_data=input_data[features]
grid_data=grid_data[features]
grid_data[['x','y']]=grid_data[['x','y']].astype(float)
#%%
ST_dict = {'K': K}
ST_ok_dict = {'Temp': Temp, 'pH': pH, 'Na': Na,'Ca': Ca,
              'Mg': Mg, 'Cl': Cl, 'SO4': SO4, 'HCO3': HCO3, 'anion': anion}
#%%
for key in ST_dict.keys():
    trn = ST_dict[key].drop(columns=key)
    tst = ST_dict[key][key]
    trn=pd.concat([trn,tst],axis=1)
    # model_s=Stacking_train_model(trn,key)
    # input_data[key]=Stacking_est(model_s,trn,key,input_data[features])
    # grid_data[key]=Stacking_est(model_s,trn,key,grid_data[features])
    input_data[key]=Stacking_model(trn,key,input_data[features])
    grid_data[key]=Stacking_model(trn,key,grid_data[features])
    
for key in ST_ok_dict.keys():
    trn = ST_ok_dict[key].drop(columns=key)
    tst = ST_ok_dict[key][key]
    trn=pd.concat([trn,tst],axis=1)
    # model_s=Stacking_train_model_ok(trn,key,parameters[key])
    # input_data[key]=Stacking_est_ok(model_s,trn,key,input_data[features],parameters[key])
    # grid_data[key]=Stacking_est_ok(model_s,trn,key,grid_data[features],parameters[key])
    input_data[key]=Stacking_model_ok(trn,key,input_data[features],parameters[key])
    grid_data[key]=Stacking_model_ok(trn,key,grid_data[features],parameters[key])
#%%
# RF_dict={'Temp':Temp,'pH':pH,'Na':Na,'anion':anion}
# OK_dict={'K':K,'Mg':Mg,'Cl':Cl,'SO4':SO4,'HCO3':HCO3}
# LGBM_dict={'Ca':Ca}
# input_data.columns
# #%%
# for key in RF_dict.keys():
#     trn = RF_dict[key].drop(columns=key)
#     tst = RF_dict[key][key]
#     rf=RandomForestRegressor()
#     rf.fit(trn,tst)
#     input_data[key]=rf.predict(input_data[features])
#     grid_data[key]=rf.predict(grid_data[features])

# for key in OK_dict.keys():
#     trn = OK_dict[key].drop(columns=key)
#     tst = OK_dict[key][key]
#     ok = OrdinaryKriging(trn['x'], trn['y'], tst, variogram_model=parameters[key]
#                                 [1], variogram_parameters=parameters[key][0])
#     input_data[key]= ok.execute('points', input_data['x'], input_data['y'])[0].data
#     grid_data[key]= ok.execute('points', grid_data['x'], grid_data['y'])[0].data

# for key in LGBM_dict.keys():
#     trn = LGBM_dict[key].drop(columns=key)
#     tst = LGBM_dict[key][key]
#     lgbm=LGBMRegressor()
#     lgbm.fit(trn,tst)
#     input_data[key]=lgbm.predict(input_data[features])
#     grid_data[key]=lgbm.predict(grid_data[features])

#%%
input_data=input_data[['x','y','Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3','anion']]
grid_data=grid_data[['x','y','Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3','anion']]
input_data.to_csv('./input_japan/onsen/add_input_onsen_ja.csv',index=False)
grid_data.to_csv('./input_japan/onsen/add_grid_onsen_detail_ja.csv',index=False)
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


if 1:
    variogram(Temp, sep=1000, max_dist=200001, parameters={
        'sill': 300, 'range': 75000, 'nugget': 175}, how='spherical')
    variogram(pH, sep=1000, max_dist=20001, parameters={
        'sill': 1.75-0.5, 'range': 2500, 'nugget': 0.5})
    variogram(Na, sep=1000, max_dist=40001, parameters={
        'sill': 2250000-800000, 'range': 7500, 'nugget': 800000}, how='spherical')
    variogram(K, sep=5000, max_dist=100001, parameters={
        'sill': 60000, 'range': 25000, 'nugget': 20000})
    variogram(Ca, sep=50, max_dist=1001, parameters={
        'sill': 150000-50000, 'range': 150, 'nugget': 50000}, how='spherical')
    variogram(Mg, sep=11000, max_dist=400001, parameters={
        'sill': 16000, 'range': 180000, 'nugget': 12000})
    variogram(Cl, sep=1000, max_dist=30001, parameters={
        'sill': 9000000-3500000, 'range': 8000, 'nugget': 3500000}, how='spherical')
    variogram(SO4, sep=1000, max_dist=25001, parameters={
        'sill': 220000, 'range': 2000, 'nugget': 80000})
    variogram(HCO3, sep=11000, max_dist=400001, parameters={
        'sill': 500000, 'range': 50000, 'nugget': 200000})
    variogram(anion, sep=1000, max_dist=20001, parameters={
        'sill': 0.03-0.007, 'range': 5000, 'nugget': 0.007})#'sill': 0.03, 'range': 50000, 'nugget': 0.02
# %%
parameters = {
    'Temp': [{'sill': 200, 'range': 50000, 'nugget': 250}, 'gaussian'],
    'pH': [{'sill': 1, 'range': 15000, 'nugget': 1.1}, 'gaussian'],
    'Na': [{'sill': 1700000, 'range': 120000, 'nugget': 2000000}, 'spherical'],
    'K': [{'sill': 60000, 'range': 25000, 'nugget': 20000}, 'gaussian'],
    'Ca': [{'sill': 55000, 'range': 10000, 'nugget': 145000}, 'gaussian'],
    'Mg': [{'sill': 16000, 'range': 180000, 'nugget': 12000}, 'gaussian'],
    'Cl': [{'sill': 6000000, 'range': 100000, 'nugget': 6500000}, 'spherical'],
    'SO4': [{'sill': 220000, 'range': 2000, 'nugget': 80000}, 'gaussian'],
    'HCO3': [{'sill': 500000, 'range': 50000, 'nugget': 200000}, 'gaussian'],
    'anion': [{'sill': 0.03-0.007, 'range': 5000, 'nugget': 0.007}, 'gaussian']
}
# %%
onsen_data = [Temp, pH, Na, K, Ca, Mg, Cl, SO4, HCO3, anion]
l = len(onsen_data)
input_data = pd.read_csv('./input/useful_data/input_data.csv').groupby(
    ['x', 'y'], as_index=False).mean()[['x', 'y']]
est_data = pd.read_csv('./input/useful_data/est_grid_500.csv').groupby(
    ['x', 'y'], as_index=False).mean()[['x', 'y']]
# %%
for i, value in enumerate(tqdm(parameters.keys())):
    if value=='HCO3' or value=='anion' or value=='Ca':
        ok = OrdinaryKriging(onsen_data[i].iloc[:, 0], onsen_data[i].iloc[:, 1], onsen_data[i].iloc[:, 2],
                             variogram_model=parameters[value][1], variogram_parameters=parameters[value][0])
    else:
        ok = OrdinaryKriging(onsen_data[i].iloc[:, 0], onsen_data[i].iloc[:, 1], onsen_data[i].iloc[:, 2])

    est_data_f = est_data.astype(float)
    input_data[value] = ok.execute(
        'points', input_data['x'], input_data['y'])[0].data
    est_data[value] = ok.execute(
        'points', est_data_f['x'], est_data_f['y'])[0].data
# %%
# input_data.to_csv('./input/onsen/add_input_onsen_new.csv',index=False)
# est_data.to_csv('./input/onsen/add_grid_onsen_new.csv',index=False)
# %%

# %%
