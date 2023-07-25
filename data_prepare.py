#%%
import pandas as pd 
import numpy as np

#%%
data=pd.read_csv('./input/坑井温度全まとめ.csv')
data.head()
# %%
# boring=data.groupby(['ido','keido'],as_index=False).mean()[['ido','keido']]
# boring.to_csv('./成形前データ/boring1.csv',index=False)
# %%
wgs_albers=pd.read_csv('./input/WGS_albers.csv')
wgs_albers
# %%
# data=data.merge(wgs_albers,how='left',on=['ido','keido'])
# data.head()
# %%
# data['h_z']=data['h']-data['z']
# data
# %%
# data=data.drop_duplicates()
# data
# %%
features=['x','y','h','z','h_z','t']
data=data[features]
# data[features].to_csv('./input/useful_data/input_data.csv',index=False)
# %%
data.head()
# %%
