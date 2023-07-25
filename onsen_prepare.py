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

seed = 0
np.random.seed(seed)
random.seed(seed)

# %%
with codecs.open(rf'./database/温泉/ONSEN.csv', 'r', 'shift-jis', 'ignore') as f:
    onsen = pd.read_csv(f)

# %%
# onsen_ido_keido=onsen.loc[:,['Latitude','Longitude']]
# onsen_ido_keido.columns=['ido','keido']
# onsen_ido_keido.to_csv('./database/温泉/onsen_ido_keido.csv',index=False)
# %%
onsen_WGS_albers = pd.read_csv('./database/温泉/onsen_WGS_albers.csv')
onsen_WGS_albers
# %%
onsen = onsen.rename(columns={'Latitude': 'ido', 'Longitude': 'keido'})
onsen = onsen.merge(onsen_WGS_albers, how='left', on=['ido', 'keido'])
onsen = onsen.rename(columns={'POINT_X': 'x', 'POINT_Y': 'y'})
onsen
# %%
# start_time=time.time()
# for ido, keido in onsen[['ido', 'keido']].values:
#     url = f'https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php?lon={keido}&lat={ido}&outtype=JSON'
#     res = requests.get(url)
#     onsen.loc[((onsen['ido'] == ido) & (onsen['keido'] == keido)),
#               'h'] = json.loads(res.text)['elevation']
#     time.sleep(3)
# onsen.to_csv('./input/onsen/database/onsen_xyh.csv',index=False)
# print(f'END {time.time()-start_time}')
# %%


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


# %%
onsen = pd.read_csv('./input/onsen/database/onsen_xyh.csv')
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

add_onsen_volcano=pd.read_csv('./input/onsen/add_onsen_volcano.csv')
add_onsen_curie=pd.read_csv('./input/onsen/add_onsen_curie.csv')
add_onsen_volcano['x'], add_onsen_volcano['y'] = add_onsen_volcano['x'].round(5), add_onsen_volcano['y'].round(5)
add_onsen_curie['x'], add_onsen_curie['y'] = add_onsen_curie['x'].round(5), add_onsen_curie['y'].round(5)
onsen_features=onsen_features.merge(add_onsen_volcano,on=['x','y'],how='left')
onsen_features=onsen_features.merge(add_onsen_curie,on=['x','y'],how='left')
onsen_features.columns
# onsen_features[onsen_features['Litho_EN']=='-----']
# onsen_features.astype(float)
# %%
onsen_features
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
anion_SO4 = anion['SO4'].values
anion_Cl = anion['Cl'].values
anion_HCO3 = anion['HCO3'].values
anion_index = 0.5*(anion_SO4/(anion_Cl+anion_SO4) +
                   (anion_Cl+anion_SO4)/(anion_Cl+anion_SO4+anion_HCO3))
anion['anion'] = anion_index
anion = anion.loc[:, ['x', 'y', 'anion']]
anion = anion.merge(onsen_features, on=['x', 'y'], how='left').dropna()
parameters = {
    'Temp': [{'sill': 200, 'range': 50000, 'nugget': 250}, 'gaussian'],
    'pH': [{'sill': 1, 'range': 15000, 'nugget': 1.1}, 'gaussian'],
    'Na': [{'sill': 1700000, 'range': 120000, 'nugget': 2000000}, 'spherical'],
    'K': [{'sill': 60000, 'range': 25000, 'nugget': 20000}, 'gaussian'],
    'Ca': [{'sill': 70000, 'range': 10000, 'nugget': 130000}, 'gaussian'],
    'Mg': [{'sill': 16000, 'range': 180000, 'nugget': 12000}, 'gaussian'],
    'Cl': [{'sill': 6500000, 'range': 100000, 'nugget': 6000000}, 'spherical'],
    'SO4': [{'sill': 220000, 'range': 2000, 'nugget': 80000}, 'gaussian'],
    'HCO3': [{'sill': 500000, 'range': 50000, 'nugget': 200000}, 'gaussian'],
    'anion': [{'sill': 0.03-0.007, 'range': 5000, 'nugget': 0.007}, 'gaussian']#'sill': 0.03, 'range': 50000, 'nugget': 0.02
}
# %%
result_df=[]
onsen_dict = {'Temp': Temp, 'pH': pH, 'Na': Na, 'K': K, 'Ca': Ca,
              'Mg': Mg, 'Cl': Cl, 'SO4': SO4, 'HCO3': HCO3, 'anion': anion}
for key in onsen_dict.keys():
    result_dict={}
    trn = onsen_dict[key].drop(columns=key)
    tst = onsen_dict[key][key]
    print(f'{key}', '-'*10, tst.std())
    MLA = {'xgb': XGBRegressor(), 'lgbm': LGBMRegressor(), 'lr': LinearRegression(
    ),  'dt': DecisionTreeRegressor(), 'rf': RandomForestRegressor(), 'kn': KNeighborsRegressor()}
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
        ok2 = OrdinaryKriging(trn_x['x'], trn_x['y'], trn_y)
        predict1 = ok1.execute('points', val_x['x'], val_x['y'])[0].data
        predict2 = ok2.execute('points', val_x['x'], val_x['y'])[0].data
        rmse_list1.append(np.sqrt(mean_squared_error(val_y.values, predict1)))
        rmse_list2.append(np.sqrt(mean_squared_error(val_y.values, predict2)))

    print('ok1', np.mean(rmse_list1))
    print('ok2', np.mean(rmse_list2))
    result_dict['ok1']=np.mean(rmse_list1)
    result_dict['ok2']=np.mean(rmse_list2)
    result_dict=pd.DataFrame(result_dict.values(),index=result_dict.keys(),columns=[key])
    # print(result_dict)
    result_df.append(result_dict)
result_df=pd.concat(result_df,axis=1)
# result_df.to_csv('./output/onsen_pred_result.csv')
# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
result_df=pd.read_csv('./output/onsen_pred_result.csv').rename(columns={'Unnamed: 0':'Model'})
s=result_df.iloc[[-1,-2],:].min()
s.name=8
result_df=result_df.append(s)
result_df=result_df.drop(index=[6,7])
result_df=result_df.reset_index(drop=True)
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
    bar=ax.bar(['XGB','LGBM','LR','DT','RF','KN','OK'],result_df[key])
    bar[result_df[key].idxmin()].set_color("red")
    plt.show()
#%%
def preprocess_input(df):
    add_input_volcano = pd.read_csv('./input/volcano/add_input_volcano.csv')
    add_input_curie = pd.read_csv('./input/curie_point/add_input_curie.csv')
    add_input_tishitsu = pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
    
    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])

    return df


def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('./input/volcano/add_grid_volcano.csv')
    add_grid_curie = pd.read_csv('./input/curie_point/add_grid_curie.csv')
    add_grid_tishitsu = pd.read_csv('./input/tishitsu/add_grid_tishitsu_pred.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])

    return df

input_data=pd.read_csv('./input/useful_data/input_data.csv').groupby(['x','y'],as_index=False).mean()[['x','y','h']]
grid_data=pd.read_csv('./input/useful_data/est_grid_500.csv').groupby(['x','y'],as_index=False).mean()[['x','y','h']]
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
RF_dict={'Temp':Temp,'pH':pH,'Na':Na,'Cl':Cl,'anion':anion}
OK_dict={'K':K,'Mg':Mg,'SO4':SO4,'HCO3':HCO3}
LGBM_dict={'Ca':Ca}
input_data.columns
#%%
for key in RF_dict.keys():
    trn = RF_dict[key].drop(columns=key)
    tst = RF_dict[key][key]
    rf=RandomForestRegressor()
    rf.fit(trn,tst)
    input_data[key]=rf.predict(input_data[features])
    grid_data[key]=rf.predict(grid_data[features])

for key in OK_dict.keys():
    trn = OK_dict[key].drop(columns=key)
    tst = OK_dict[key][key]
    ok = OrdinaryKriging(trn['x'], trn['y'], tst, variogram_model=parameters[key]
                                [1], variogram_parameters=parameters[key][0])
    input_data[key]= ok.execute('points', input_data['x'], input_data['y'])[0].data
    grid_data[key]= ok.execute('points', grid_data['x'], grid_data['y'])[0].data

for key in LGBM_dict.keys():
    trn = LGBM_dict[key].drop(columns=key)
    tst = LGBM_dict[key][key]
    lgbm=LGBMRegressor()
    lgbm.fit(trn,tst)
    input_data[key]=lgbm.predict(input_data[features])
    grid_data[key]=lgbm.predict(grid_data[features])

#%%
input_data=input_data[['x','y','Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3','anion']]
grid_data=grid_data[['x','y','Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3','anion']]
# input_data.to_csv('./input/onsen/add_input_onsen.csv',index=False)
# grid_data.to_csv('./input/onsen/add_grid_onsen.csv',index=False)

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
    plt.show()


if 0:
    variogram(Temp, sep=11000, max_dist=300001, parameters={
        'sill': 200, 'range': 50000, 'nugget': 250})
    variogram(pH, sep=5000, max_dist=100001, parameters={
        'sill': 1, 'range': 15000, 'nugget': 1.1})
    variogram(Na, sep=21000, max_dist=500001, parameters={
        'sill': 1700000, 'range': 120000, 'nugget': 2000000}, how='spherical')
    variogram(K, sep=5000, max_dist=100001, parameters={
        'sill': 60000, 'range': 25000, 'nugget': 20000})
    variogram(Ca, sep=5000, max_dist=100001, parameters={
        'sill': 70000, 'range': 10000, 'nugget': 130000})
    variogram(Mg, sep=11000, max_dist=400001, parameters={
        'sill': 16000, 'range': 180000, 'nugget': 12000})
    variogram(Cl, sep=11000, max_dist=300001, parameters={
        'sill': 6500000, 'range': 100000, 'nugget': 6000000}, how='spherical')
    variogram(SO4, sep=1000, max_dist=25001, parameters={
        'sill': 220000, 'range': 2000, 'nugget': 80000})
    variogram(HCO3, sep=11000, max_dist=400001, parameters={
        'sill': 500000, 'range': 50000, 'nugget': 200000})
    variogram(anion, sep=11000, max_dist=500001, parameters={
        'sill': 0.03-0.007, 'range': 5000, 'nugget': 0.007})
# %%
parameters = {
    'Temp': [{'sill': 200, 'range': 50000, 'nugget': 250}, 'gaussian'],
    'pH': [{'sill': 1, 'range': 15000, 'nugget': 1.1}, 'gaussian'],
    'Na': [{'sill': 1700000, 'range': 120000, 'nugget': 2000000}, 'spherical'],
    'K': [{'sill': 60000, 'range': 25000, 'nugget': 20000}, 'gaussian'],
    'Ca': [{'sill': 70000, 'range': 10000, 'nugget': 130000}, 'gaussian'],
    'Mg': [{'sill': 16000, 'range': 180000, 'nugget': 12000}, 'gaussian'],
    'Cl': [{'sill': 6500000, 'range': 100000, 'nugget': 6000000}, 'spherical'],
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
