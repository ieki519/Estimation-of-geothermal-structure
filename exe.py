#%%
# if 0:#grid
#     import pandas as pd
#     import re
#     import requests
#     import time
#     import json
#     xy_grid_wgs=pd.read_csv('./input/elevation/hyoukou_wgs.csv')[['ido','keido']]
#     all_tishitsu=[]
#     time_now=time.time()
#     for ido,keido in xy_grid_wgs.values:
#         url=f'https://gbank.gsj.jp/seamless/v2/api/1.2.1/legend.json?point={ido},{keido}'
#         res=requests.get(url)
#         tishitsu=res.text
#         dict_tishitsu=json.loads(tishitsu)
#         df_tishitsu=pd.DataFrame(dict_tishitsu.values(),index=dict_tishitsu.keys()).T
#         df_tishitsu['ido'],df_tishitsu['keido']=ido,keido
#         df_tishitsu=df_tishitsu[['ido','keido','symbol','formationAge_ja','group_ja','lithology_ja']]
#         all_tishitsu.append(df_tishitsu)
#         time.sleep(5)
#     all_tishitsu=pd.concat(all_tishitsu).reset_index(drop=True)
#     all_tishitsu.to_csv('./input/tishitsu/add_grid_tishitsu_idokeido.csv',index=False)

#     all_tishitsu.loc[:,['ido','keido']]=all_tishitsu[['ido','keido']].round(3)
#     grid_wgs_albers=pd.read_csv('./input/WGS_to_albers/grid_WGS_albers.csv')
#     grid_wgs_albers['ido'],grid_wgs_albers['keido']=grid_wgs_albers['ido'].round(3),grid_wgs_albers['keido'].round(3)
#     all_tishitsu=all_tishitsu.merge(grid_wgs_albers,how='left',on=['ido','keido'])
#     all_tishitsu=all_tishitsu[['x','y','symbol','formationAge_ja','group_ja','lithology_ja']]
#     all_tishitsu.to_csv('./input/tishitsu/add_grid_tishitsu_.csv',index=False)
#     print((time.time()-time_now))
#%%
# if 0:#input
#     import pandas as pd
#     import re
#     import requests
#     import time
#     import json
#     xy_input_wgs=pd.read_csv('./input/WGS_to_albers/input_WGS_albers.csv')[['ido','keido']]
#     all_tishitsu=[]
#     time_now=time.time()
#     for ido,keido in xy_input_wgs.values:
#         url=f'https://gbank.gsj.jp/seamless/v2/api/1.2.1/legend.json?point={ido},{keido}'
#         res=requests.get(url)
#         tishitsu=res.text
#         dict_tishitsu=json.loads(tishitsu)
#         df_tishitsu=pd.DataFrame(dict_tishitsu.values(),index=dict_tishitsu.keys()).T
#         df_tishitsu['ido'],df_tishitsu['keido']=ido,keido
#         df_tishitsu=df_tishitsu[['ido','keido','symbol','formationAge_ja','group_ja','lithology_ja']]
#         all_tishitsu.append(df_tishitsu)
#         time.sleep(5)
#     all_tishitsu=pd.concat(all_tishitsu).reset_index(drop=True)
#     # all_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu_idokeido.csv',index=False)
#     #%%
#     all_tishitsu=pd.read_csv('./input/tishitsu/add_input_tishitsu_idokeido.csv')
#     all_tishitsu.loc[:,['ido','keido']]=all_tishitsu[['ido','keido']].round(8)

#     input_wgs_albers=pd.read_csv('./input/WGS_to_albers/input_WGS_albers.csv')
#     input_wgs_albers['ido'],input_wgs_albers['keido']=input_wgs_albers['ido'].round(8),input_wgs_albers['keido'].round(8)
#     input_wgs_albers=input_wgs_albers[['ido','keido','x','y']]

#     all_tishitsu=all_tishitsu.merge(input_wgs_albers,on=['ido','keido'],how='left')
#     all_tishitsu=all_tishitsu[['x','y','symbol','formationAge_ja','group_ja','lithology_ja']]
#     # all_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu.csv',index=False)
#     print((time.time()-time_now))

# add_input_tishitsu=pd.read_csv('./input/tishitsu/add_input_tishitsu.csv')
# add_input_tishitsu['x'],add_input_tishitsu['y']=add_input_tishitsu['x'].round(),add_input_tishitsu['y'].round()
# add_input_tishitsu=add_input_tishitsu.drop_duplicates().reset_index(drop=True)
# add_input_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu.csv',index=False)
# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import folium
#%%
# add_input_tishitsu=pd.read_csv('./input/tishitsu/database/add_input_tishitsu.csv')
# add_grid_tishitsu=pd.read_csv('./input/tishitsu/database/add_grid_tishitsu.csv')
add_grid_tishitsu_detail=pd.read_csv('./input/tishitsu/database/add_grid_tishitsu_detail_.csv')
add_grid_tishitsu_detail
# f = folium.Figure(width=1000, height=500)
# f = folium.Figure(width=1000, height=500)
# center_lat=34.686567
# center_lon=135.52000
# m = folium.Map([center_lat,center_lon], zoom_start=4).add_to(f)
# for ido,keido in add_grid_tishitsu[add_grid_tishitsu['symbol'].isnull()][['ido','keido']].values:
#     folium.Marker(location=[ido,keido]).add_to(m)
# m
#%%
input_data=pd.read_csv('./input/useful_data/input_data.csv').groupby(['x','y'],as_index=False).mean()[['x','y','h']]
est_grid=pd.read_csv('./input/useful_data/est_grid_500.csv').groupby(['x','y'],as_index=False).mean()[['x','y','h']]
add_input_tishitsu=add_input_tishitsu.merge(input_data,how='left',on=['x','y'])
add_grid_tishitsu=add_grid_tishitsu.merge(est_grid,how='left',on=['x','y'])
tishitsu=pd.concat([add_input_tishitsu,add_grid_tishitsu])

train_tishitsu=tishitsu[~tishitsu['symbol'].isnull()]
test_input_tishitsu=add_input_tishitsu[add_input_tishitsu['symbol'].isnull()]
test_grid_tishitsu=add_grid_tishitsu[add_grid_tishitsu['symbol'].isnull()]

trn,tst=train_test_split(train_tishitsu,test_size=0.1,shuffle=True)

for target in ['symbol','formationAge_ja', 'group_ja', 'lithology_ja']:
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(trn.loc[:,['x','y','h']],trn[target])
    print(accuracy_score(tst[target],knn.predict(tst.loc[:,['x','y','h']])))

for target in ['symbol','formationAge_ja', 'group_ja', 'lithology_ja']:
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_tishitsu.loc[:,['x','y','h']],train_tishitsu[target])
    test_input_tishitsu.loc[:,target]=knn.predict(test_input_tishitsu.loc[:,['x','y','h']])
    test_grid_tishitsu.loc[:,target]=knn.predict(test_grid_tishitsu.loc[:,['x','y','h']])

add_input_tishitsu=pd.concat([add_input_tishitsu.dropna(),test_input_tishitsu]).reset_index(drop=True)
add_grid_tishitsu=pd.concat([add_grid_tishitsu.dropna(),test_grid_tishitsu]).reset_index(drop=True)

add_input_tishitsu=add_input_tishitsu[['x', 'y','symbol','formationAge_ja', 'group_ja', 'lithology_ja']]
add_grid_tishitsu=add_grid_tishitsu[['x', 'y','symbol','formationAge_ja', 'group_ja', 'lithology_ja']]

# add_input_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu_pred.csv',index=False)
# add_grid_tishitsu.to_csv('./input/tishitsu/add_grid_tishitsu_pred.csv',index=False)

# %%


add_input_tishitsu=pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
add_grid_tishitsu=pd.read_csv('./input/tishitsu/add_grid_tishitsu_pred.csv')
tishitsu=pd.concat([add_input_tishitsu,add_grid_tishitsu])
# pd.DataFrame(np.sort(tishitsu['formationAge_ja'].unique())).to_csv('./成形前データ/地質年代.csv',index=False,encoding='cp932')
# %%
import codecs
with codecs.open(rf'./input/tishitsu/地質年代.csv','r','shift-jis','ignore') as f:
    age=pd.read_csv(f)
age
# %%
add_input_tishitsu=add_input_tishitsu.merge(age,on=['formationAge_ja'],how='left')
add_grid_tishitsu=add_grid_tishitsu.merge(age,on=['formationAge_ja'],how='left')
add_grid_tishitsu
# %%
def encode_top_100(s):#Seriesを入力
    uniqs,freqs=np.unique(s,return_counts=True)
    top=sorted(zip(uniqs,freqs),key=lambda x:x[1],reverse=True)
    top_map={uf[0]:lank for uf,lank in zip(top,range(len(top)))}
    return s.map(lambda x:top_map.get(x,0)).astype(np.int),top_map
# %%
add_grid_tishitsu['symbol_freq'],freq_dict=encode_top_100(add_grid_tishitsu['symbol'])
add_input_tishitsu['symbol_freq']=add_input_tishitsu['symbol'].map(lambda x:freq_dict.get(x,0))
add_grid_tishitsu['group_freq'],freq_dict=encode_top_100(add_grid_tishitsu['group_ja'])
add_input_tishitsu['group_freq']=add_input_tishitsu['group_ja'].map(lambda x:freq_dict.get(x,0))
add_grid_tishitsu['lithology_freq'],freq_dict=encode_top_100(add_grid_tishitsu['lithology_ja'])
add_input_tishitsu['lithology_freq']=add_input_tishitsu['lithology_ja'].map(lambda x:freq_dict.get(x,0))
# add_input_tishitsu.to_csv('./input/tishitsu/add_input_tishitsu_pred.csv',index=False)
# add_grid_tishitsu.to_csv('./input/tishitsu/add_grid_tishitsu_pred.csv',index=False)
# %%
# %%
import seaborn as sns
#%%
add_grid_tishitsu['lithology_ja'].unique().shape
# %%
sns.countplot(x='lithology_ja',data=add_input_tishitsu)
# %%
