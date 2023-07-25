#%%
# import pandas as pd
# %%
# prepare h_z
# grid_detail=pd.read_csv("./成形前データ/grid_detail.csv")
# hyoukou_detail=pd.read_csv("./input/elevation/hyoukou_albers_detail.csv")[["x","y","hyoukou"]]
# grid_detail=grid_detail.merge(hyoukou_detail,on=["x","y"],how="left")
# grid_detail=grid_detail.dropna().reset_index()
# del grid_detail["index"]
# grid_detail=grid_detail.rename(columns={"hyoukou":"h"})
# grid_detail=grid_detail.rename(columns={"z":"h_z"})
# grid_detail["z"]=grid_detail["h"]-grid_detail["h_z"]
# grid_detail[["x","y","h","z","h_z"]].to_csv("./input/useful_data/est_grid_detail.csv",index=False)
# %%
#%%
#prepare volcano
# import pandas as pd
# import numpy as np
# from scipy.spatial.distance import pdist, squareform,cdist
# #%%
# #grid
# #%%
# volcano=pd.read_csv('./input/volcano/volcano.csv')
# volcano=volcano.values
# volcano
# #%%
# xy_grid_albers=pd.read_csv('./input/useful_data/est_grid_detail.csv')
# xy_grid_albers=xy_grid_albers.groupby(['x','y'],as_index=False).mean()[['x','y']].values
# xy_grid_albers.shape
# # %%
# volcano_dist=cdist(xy_grid_albers,volcano)
# # %%
# volcano_dist
# # %%
# volcano_dist_list=[]
# for i in range(volcano_dist.shape[0]):
#     volcano_dist_list.append(np.min(volcano_dist[i,:]))
# volcano_dist_list
# # %%
# xy_grid_albers_master=pd.read_csv('./input/useful_data/est_grid_detail.csv')
# xy_grid_albers_master=xy_grid_albers_master.groupby(['x','y'],as_index=False).mean()[['x','y']]
# xy_grid_albers_master['volcano']=volcano_dist_list
# xy_grid_albers_master
# %%
# xy_grid_albers_master.to_csv('./input/volcano/add_grid_volcano_detail.csv',index=False)
# %%
#%%
# prepare curie
# import pandas as pd
# import numpy as np
# from pykrige.ok import OrdinaryKriging
# from scipy.spatial.distance import pdist, squareform,cdist
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import pykrige.variogram_models as vm
# #%%
# ##feature
# curie=pd.read_csv('./input/curie_point/database/curie_albers.csv')
# curie
# #%%
# xy_dis=squareform(pdist(curie[['x','y']]))
# #%%
# t_vario=squareform(pdist(curie['curie'].values.reshape(-1,1))**2)
# # %%
# sep=1100
# max_dist=200001
# sv_i=np.zeros(len(range(0,max_dist,sep)))
# for i,value in enumerate(tqdm(range(0,max_dist,sep))):
#     mask1=xy_dis>value
#     mask2=xy_dis<value+sep
#     mask=mask1*mask2
#     res1=t_vario[mask]
#     mask3=res1>0
#     res2=(res1[mask3].mean())/2
#     sv_i[i]=res2
# sv_i
# # %%
# x=range(0,max_dist,sep)
# plt.plot(x[:],sv_i[:],c='black',marker='o')
# plt.plot(x[:],vm.spherical_variogram_model([10,200000,0.5],np.array(x))[:],c='red')

# #%%

# curie=curie.values
# okd = OrdinaryKriging(curie[:, 0], curie[:, 1], curie[:, 2],
#                                                  variogram_model='spherical',variogram_parameters={'sill':10,'range':200000,'nugget':0.5})  
# curie
# #%%
# est_data = pd.read_csv('./input/useful_data/est_grid_detail.csv')
# est_data=est_data.groupby(['x','y'],as_index=False).mean()[['x','y']]
# est_xy=est_data.values.astype('float')
# input_predict=okd.execute('points',est_xy[:, 0], est_xy[:, 1])[0].data
# est_data['curie']=input_predict
# est_data.to_csv('./input/curie_point/add_grid_curie_detail.csv',index=False)
# #%%
# #%%
# prepare tishitsu
# import pandas as pd
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.utils import shuffle
# import folium
# #%%
# add_input_tishitsu=pd.read_csv('./input/tishitsu/database/add_input_tishitsu.csv')
# add_grid_tishitsu=pd.read_csv('./input/tishitsu/database/add_grid_tishitsu_detail_.csv')

# # f = folium.Figure(width=1000, height=500)
# # f = folium.Figure(width=1000, height=500)
# # center_lat=34.686567
# # center_lon=135.52000
# # m = folium.Map([center_lat,center_lon], zoom_start=4).add_to(f)
# # for ido,keido in add_grid_tishitsu[add_grid_tishitsu['symbol'].isnull()][['ido','keido']].values:
# #     folium.Marker(location=[ido,keido]).add_to(m)
# # m
# #%%
# input_data=pd.read_csv('./input/useful_data/input_data.csv').groupby(['x','y'],as_index=False).mean()[['x','y','h']]
# est_grid=pd.read_csv('./input/useful_data/est_grid_detail.csv').groupby(['x','y'],as_index=False).mean()[['x','y','h']]
# add_input_tishitsu=add_input_tishitsu.merge(input_data,how='left',on=['x','y'])
# add_grid_tishitsu=add_grid_tishitsu.merge(est_grid,how='left',on=['x','y'])
# tishitsu=pd.concat([add_input_tishitsu,add_grid_tishitsu])

# train_tishitsu=tishitsu[~tishitsu['symbol'].isnull()]
# test_input_tishitsu=add_input_tishitsu[add_input_tishitsu['symbol'].isnull()]
# test_grid_tishitsu=add_grid_tishitsu[add_grid_tishitsu['symbol'].isnull()]

# trn,tst=train_test_split(train_tishitsu,test_size=0.1,shuffle=True)

# for target in ['symbol','formationAge_ja', 'group_ja', 'lithology_ja']:
#     knn=KNeighborsClassifier(n_neighbors=1)
#     knn.fit(trn.loc[:,['x','y','h']],trn[target])
#     print(accuracy_score(tst[target],knn.predict(tst.loc[:,['x','y','h']])))

# for target in ['symbol','formationAge_ja', 'group_ja', 'lithology_ja']:
#     knn=KNeighborsClassifier(n_neighbors=1)
#     knn.fit(train_tishitsu.loc[:,['x','y','h']],train_tishitsu[target])
#     test_input_tishitsu.loc[:,target]=knn.predict(test_input_tishitsu.loc[:,['x','y','h']])
#     test_grid_tishitsu.loc[:,target]=knn.predict(test_grid_tishitsu.loc[:,['x','y','h']])

# add_input_tishitsu=pd.concat([add_input_tishitsu.dropna(),test_input_tishitsu]).reset_index(drop=True)
# add_grid_tishitsu=pd.concat([add_grid_tishitsu.dropna(),test_grid_tishitsu]).reset_index(drop=True)

# add_input_tishitsu=add_input_tishitsu[['x', 'y','symbol','formationAge_ja', 'group_ja', 'lithology_ja']]
# add_grid_tishitsu=add_grid_tishitsu[['x', 'y','symbol','formationAge_ja', 'group_ja', 'lithology_ja']]

# # add_input_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu_pred.csv',index=False)
# # add_grid_tishitsu.to_csv('./input/tishitsu/add_grid_tishitsu_detail_pred.csv',index=False)

# # %%

# import codecs
# with codecs.open(rf'./成形前データ/地質年代.csv','r','shift-jis','ignore') as f:
#     df=pd.read_csv(f)

# #%%
# l=[]
# add_input_tishitsu=pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
# add_grid_tishitsu=pd.read_csv('./input/tishitsu/add_grid_tishitsu_detail_pred.csv')
# tishitsu=pd.concat([add_input_tishitsu,add_grid_tishitsu])
# # pd.DataFrame(np.sort(tishitsu['formationAge_ja'].unique())).to_csv('./成形前データ/地質年代.csv',index=False,encoding='cp932')
# # %%
# import codecs
# with codecs.open(rf'./input/tishitsu/地質年代.csv','r','shift-jis','ignore') as f:
#     age=pd.read_csv(f)
# age
# # %%
# add_input_tishitsu=add_input_tishitsu.merge(age,on=['formationAge_ja'],how='left')
# add_grid_tishitsu=add_grid_tishitsu.merge(age,on=['formationAge_ja'],how='left')
# add_grid_tishitsu
# # %%
# def encode_top_100(s):#Seriesを入力
#     uniqs,freqs=np.unique(s,return_counts=True)
#     top=sorted(zip(uniqs,freqs),key=lambda x:x[1],reverse=True)
#     top_map={uf[0]:lank for uf,lank in zip(top,range(len(top)))}
#     return s.map(lambda x:top_map.get(x,0)).astype(np.int),top_map
# # %%
# add_grid_tishitsu['symbol_freq'],freq_dict=encode_top_100(add_grid_tishitsu['symbol'])
# add_input_tishitsu['symbol_freq']=add_input_tishitsu['symbol'].map(lambda x:freq_dict.get(x,0))
# add_grid_tishitsu['group_freq'],freq_dict=encode_top_100(add_grid_tishitsu['group_ja'])
# add_input_tishitsu['group_freq']=add_input_tishitsu['group_ja'].map(lambda x:freq_dict.get(x,0))
# add_grid_tishitsu['lithology_freq'],freq_dict=encode_top_100(add_grid_tishitsu['lithology_ja'])
# add_input_tishitsu['lithology_freq']=add_input_tishitsu['lithology_ja'].map(lambda x:freq_dict.get(x,0))
# # add_input_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu_pred.csv',index=False)
# add_grid_tishitsu.to_csv('./input/tishitsu/add_grid_tishitsu_detail_pred.csv',index=False)
# %%
# pd.read_csv('./input/tishitsu/add_grid_tishitsu_detail_pred.csv').dropna()
# %% # prepare onsen
# # %%
# from os import replace
# from pykrige.ok import OrdinaryKriging
# from threading import local
# import pandas as pd
# import numpy as np
# import codecs
# from scipy.spatial.distance import pdist, squareform
# from seaborn.external.docscrape import Parameter
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import pykrige.variogram_models as vm
# import seaborn as sns
# import time
# import json
# import requests
# from sklearn.model_selection import train_test_split
# import time
# import xgboost as xgb
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
# from sklearn.linear_model import SGDRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.preprocessing import StandardScaler
# from pykrige.ok import OrdinaryKriging
# from sklearn.metrics import mean_squared_error
# import pykrige.variogram_models as vm
# import random
# from sklearn import model_selection
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import StratifiedKFold, KFold

# seed = 0
# np.random.seed(seed)
# random.seed(seed)

# # %%
# with codecs.open(rf'./database/温泉/ONSEN.csv', 'r', 'shift-jis', 'ignore') as f:
#     onsen = pd.read_csv(f)

# # %%
# # onsen_ido_keido=onsen.loc[:,['Latitude','Longitude']]
# # onsen_ido_keido.columns=['ido','keido']
# # onsen_ido_keido.to_csv('./database/温泉/onsen_ido_keido.csv',index=False)
# # %%
# onsen_WGS_albers = pd.read_csv('./database/温泉/onsen_WGS_albers.csv')
# onsen_WGS_albers
# # %%
# onsen = onsen.rename(columns={'Latitude': 'ido', 'Longitude': 'keido'})
# onsen = onsen.merge(onsen_WGS_albers, how='left', on=['ido', 'keido'])
# onsen = onsen.rename(columns={'POINT_X': 'x', 'POINT_Y': 'y'})
# onsen
# # %%
# # start_time=time.time()
# # for ido, keido in onsen[['ido', 'keido']].values:
# #     url = f'https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php?lon={keido}&lat={ido}&outtype=JSON'
# #     res = requests.get(url)
# #     onsen.loc[((onsen['ido'] == ido) & (onsen['keido'] == keido)),
# #               'h'] = json.loads(res.text)['elevation']
# #     time.sleep(3)
# # onsen.to_csv('./input/onsen/database/onsen_xyh.csv',index=False)
# # print(f'END {time.time()-start_time}')
# # %%


# def encode_top(s, lim):  # Seriesを入力
#     uniqs, freqs = np.unique(s, return_counts=True)
#     top = sorted(zip(uniqs, freqs), key=lambda x: x[1], reverse=True)[:lim]
#     top_map = {uf[0]: lank for uf, lank in zip(top, range(len(top)))}
#     return s.map(lambda x: top_map.get(x, lim)).astype(np.int),top_map


# def one_hot_encode(df, col):
#     tmp = pd.get_dummies(df[col], col)
#     df = pd.concat([df, tmp], axis=1)
#     df.drop(col, axis=1, inplace=True)
#     return df


# # %%
# onsen = pd.read_csv('./input/onsen/database/onsen_xyh.csv')
# onsen = onsen[~(onsen['h'] == '-----')]

# onsen['x'], onsen['y'] = onsen['x'].round(5), onsen['y'].round(5)
# onsen_features = ['x', 'y', 'h',
#                   'Simplified_rock_category']#, 'Rock_age_EN', 'Litho_EN'
# onsen_features = onsen[onsen_features].copy()
# onsen_features['Simplified_rock_category'] = onsen_features['Simplified_rock_category'].fillna(
#     'nan')
# onsen_features[['x', 'y', 'h']] = onsen_features[[
#     'x', 'y', 'h']].astype(np.float)
# rock_cat_dict={'sediments':'堆積岩', 'volcanic':'火成岩', 'Quat sediments':'堆積岩', 'nan':'その他',
#        'accretionary complex':'付加体', 'mafic plutonic':'火成岩', 'granitic':'火成岩', 'ultramafic':'火成岩',
#        'felsic plutonic':'火成岩', 'gneiss and schist':'変成岩'}
# onsen_features['Simplified_rock_category']=onsen_features['Simplified_rock_category'].apply(lambda x:rock_cat_dict[x])
# onsen_features['Simplified_rock_category'],top_map = encode_top(onsen_features['Simplified_rock_category'], 5)
# onsen_features = one_hot_encode(onsen_features, 'Simplified_rock_category')

# add_onsen_volcano=pd.read_csv('./input/onsen/add_onsen_volcano.csv')
# add_onsen_curie=pd.read_csv('./input/onsen/add_onsen_curie.csv')
# add_onsen_volcano['x'], add_onsen_volcano['y'] = add_onsen_volcano['x'].round(5), add_onsen_volcano['y'].round(5)
# add_onsen_curie['x'], add_onsen_curie['y'] = add_onsen_curie['x'].round(5), add_onsen_curie['y'].round(5)
# onsen_features=onsen_features.merge(add_onsen_volcano,on=['x','y'],how='left')
# onsen_features=onsen_features.merge(add_onsen_curie,on=['x','y'],how='left')
# onsen_features.columns
# # onsen_features[onsen_features['Litho_EN']=='-----']
# # onsen_features.astype(float)
# # %%
# onsen_features
# # %%
# Temp = onsen.loc[:, ['x', 'y', 'Temp']].dropna().astype(float)
# pH = onsen.loc[:, ['x', 'y', 'pH']].dropna().astype(float)
# Na = onsen.loc[:, ['x', 'y', 'Na']].dropna().astype(float)
# K = onsen.loc[:, ['x', 'y', 'K']].dropna().replace(
#     ['<0.1', '<1.0'], 0).astype(float)
# Ca = onsen.loc[:, ['x', 'y', 'Ca']].dropna().replace(['<1.0'], 0).astype(float)
# Mg = onsen.loc[:, ['x', 'y', 'Mg']].dropna().replace(
#     ['<0.1', '<1.0', '<0.01', '<0.02', 'TR'], 0).astype(float)
# Cl = onsen.loc[:, ['x', 'y', 'Cl']].dropna().replace(['TR'], 0).astype(float)
# SO4 = onsen.loc[:, ['x', 'y', 'SO4']].dropna().replace(
#     ['<5', '<0.05', 'N.D.', '<0.1', '<1', '<1.0', 'TR'], 0).astype(float)
# HCO3 = onsen.loc[:, ['x', 'y', 'HCO3']].dropna().replace(
#     ['TR', 'N.D.', '<0.6', '<10', '<0.01', '<0.1'], 0).astype(float)
# # %%
# Temp = Temp.merge(onsen_features, on=['x', 'y'], how='left').dropna()
# pH = pH.merge(onsen_features, on=['x', 'y'], how='left').dropna()
# Na = Na.merge(onsen_features, on=['x', 'y'], how='left').dropna()
# K = K.merge(onsen_features, on=['x', 'y'], how='left').dropna()
# Ca = Ca.merge(onsen_features, on=['x', 'y'], how='left').dropna()
# Mg = Mg.merge(onsen_features, on=['x', 'y'], how='left').dropna()
# Cl = Cl.merge(onsen_features, on=['x', 'y'], how='left').dropna()
# SO4 = SO4.merge(onsen_features, on=['x', 'y'], how='left').dropna()
# HCO3 = HCO3.merge(onsen_features, on=['x', 'y'], how='left').dropna()
# # %%
# anion = onsen.loc[:, ['x', 'y', 'SO4', 'Cl', 'HCO3']].dropna().replace(
#     ['<5', '<0.05', 'N.D.', '<0.1', '<1', '<1.0', 'TR', '<0.6', '<10', '<0.01'], 0).astype(float)
# anion_SO4 = anion['SO4'].values
# anion_Cl = anion['Cl'].values
# anion_HCO3 = anion['HCO3'].values
# anion_index = 0.5*(anion_SO4/(anion_Cl+anion_SO4) +
#                    (anion_Cl+anion_SO4)/(anion_Cl+anion_SO4+anion_HCO3))
# anion['anion'] = anion_index
# anion = anion.loc[:, ['x', 'y', 'anion']]
# anion = anion.merge(onsen_features, on=['x', 'y'], how='left').dropna()
# parameters = {
#     'Temp': [{'sill': 200, 'range': 50000, 'nugget': 250}, 'gaussian'],
#     'pH': [{'sill': 1, 'range': 15000, 'nugget': 1.1}, 'gaussian'],
#     'Na': [{'sill': 1700000, 'range': 120000, 'nugget': 2000000}, 'spherical'],
#     'K': [{'sill': 60000, 'range': 25000, 'nugget': 20000}, 'gaussian'],
#     'Ca': [{'sill': 70000, 'range': 10000, 'nugget': 130000}, 'gaussian'],
#     'Mg': [{'sill': 16000, 'range': 180000, 'nugget': 12000}, 'gaussian'],
#     'Cl': [{'sill': 6500000, 'range': 100000, 'nugget': 6000000}, 'spherical'],
#     'SO4': [{'sill': 220000, 'range': 2000, 'nugget': 80000}, 'gaussian'],
#     'HCO3': [{'sill': 500000, 'range': 50000, 'nugget': 200000}, 'gaussian'],
#     'anion': [{'sill': 0.03, 'range': 50000, 'nugget': 0.02}, 'gaussian']
# }
# # %%
# # def preprocess_input(df):
# #     add_input_volcano = pd.read_csv('./input/volcano/add_input_volcano.csv')
# #     add_input_curie = pd.read_csv('./input/curie_point/add_input_curie.csv')
# #     add_input_tishitsu = pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
    
# #     df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
# #     df = df.merge(add_input_curie, how='left', on=['x', 'y'])
# #     df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])

# #     return df


# def preprocess_grid(df):
#     add_grid_volcano = pd.read_csv('./input/volcano/add_grid_volcano_detail.csv')
#     add_grid_curie = pd.read_csv('./input/curie_point/add_grid_curie_detail.csv')
#     add_grid_tishitsu = pd.read_csv('./input/tishitsu/add_grid_tishitsu_detail_pred.csv')

#     df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
#     df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
#     df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])

#     return df

# # input_data=pd.read_csv('./input/useful_data/input_data.csv').groupby(['x','y'],as_index=False).mean()[['x','y','h']]
# grid_data=pd.read_csv('./input/useful_data/est_grid_detail.csv').groupby(['x','y'],as_index=False).mean()[['x','y','h']]
# # input_data=preprocess_input(input_data)
# grid_data=preprocess_grid(grid_data)
# # input_data['Simplified_rock_category']=input_data['group_ja'].map(lambda x:top_map[x])
# grid_data['Simplified_rock_category']=grid_data['group_ja'].map(lambda x:top_map[x])
# # input_data = one_hot_encode(input_data, 'Simplified_rock_category')
# grid_data = one_hot_encode(grid_data, 'Simplified_rock_category')
# features=['x', 'y', 'h', 'Simplified_rock_category_0',
#        'Simplified_rock_category_1', 'Simplified_rock_category_2',
#        'Simplified_rock_category_3', 'Simplified_rock_category_4', 'volcano',
#        'curie']
# # input_data=input_data[features]
# grid_data=grid_data[features]
# grid_data[['x','y']]=grid_data[['x','y']].astype(float)
# #%%
# RF_dict={'Temp':Temp,'pH':pH,'Na':Na,'Cl':Cl,'anion':anion}
# OK_dict={'K':K,'Mg':Mg,'SO4':SO4,'HCO3':HCO3}
# LGBM_dict={'Ca':Ca}
# # input_data.columns
# #%%
# for key in RF_dict.keys():
#     trn = RF_dict[key].drop(columns=key)
#     tst = RF_dict[key][key]
#     rf=RandomForestRegressor()
#     rf.fit(trn,tst)
#     # input_data[key]=rf.predict(input_data[features])
#     grid_data[key]=rf.predict(grid_data[features])

# for key in OK_dict.keys():
#     trn = OK_dict[key].drop(columns=key)
#     tst = OK_dict[key][key]
#     ok = OrdinaryKriging(trn['x'], trn['y'], tst, variogram_model=parameters[key]
#                                 [1], variogram_parameters=parameters[key][0])
#     # input_data[key]= ok.execute('points', input_data['x'], input_data['y'])[0].data
#     grid_data[key]= ok.execute('points', grid_data['x'], grid_data['y'])[0].data

# for key in LGBM_dict.keys():
#     trn = LGBM_dict[key].drop(columns=key)
#     tst = LGBM_dict[key][key]
#     lgbm=LGBMRegressor()
#     lgbm.fit(trn,tst)
#     # input_data[key]=lgbm.predict(input_data[features])
#     grid_data[key]=lgbm.predict(grid_data[features])

# #%%
# # input_data=input_data[['x','y','Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3','anion']]
# grid_data=grid_data[['x','y','Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3','anion']]
# # input_data.to_csv('./input/onsen/add_input_onsen.csv',index=False)
# grid_data
#%%
# grid_data.to_csv('./input/onsen/add_grid_onsen_detail.csv',index=False)

# %%

# %%
