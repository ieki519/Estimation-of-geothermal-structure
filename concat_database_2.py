#%%
import pandas as pd
import numpy as np
# %%
profile_db=pd.read_csv('./input/useful_data/database/profile_db.csv')
geothermal_db=pd.read_csv('./input/useful_data/database/geothermal_db.csv')
GSJ_db=pd.read_csv('./input/useful_data/database/GSJ_db.csv').sort_values(["x","y"])
# %%
profile_db['x'],profile_db['y']=profile_db['x'].round(),profile_db['y'].round()
geothermal_db['x'],geothermal_db['y']=geothermal_db['x'].round(),geothermal_db['y'].round()
GSJ_db['x'],GSJ_db['y']=GSJ_db['x'].round(),GSJ_db['y'].round()

profile_db['round_x'],profile_db['round_y']=profile_db['x'].round(-3),profile_db['y'].round(-3)
geothermal_db['round_x'],geothermal_db['round_y']=geothermal_db['x'].round(-3),geothermal_db['y'].round(-3)
GSJ_db['round_x'],GSJ_db['round_y']=GSJ_db['x'].round(-3),GSJ_db['y'].round(-3)
#%%
xy_profile_db = profile_db.groupby(['round_x','round_y'],as_index=False).mean()[['round_x','round_y']]
#%%
for x,y in xy_profile_db.values:
    geothermal_db=geothermal_db[~((geothermal_db['round_x']==x)&(geothermal_db['round_y']==y))]
geothermal_db.groupby(['x','y'],as_index=False).mean()[['x','y']].shape
#%%
xy_geothermal_db = geothermal_db.groupby(['round_x','round_y'],as_index=False).mean()[['round_x','round_y']]
xy_geothermal_db
#%%
for x,y in xy_profile_db.values:
    GSJ_db=GSJ_db[~((GSJ_db['round_x']==x)&(GSJ_db['round_y']==y))]
for x,y in xy_geothermal_db.values:
    GSJ_db=GSJ_db[~((GSJ_db['round_x']==x)&(GSJ_db['round_y']==y))]
GSJ_db.groupby(['x','y'],as_index=False).mean()[['x','y']].shape
#%%
input_data=pd.concat([profile_db,geothermal_db,GSJ_db]).reset_index(drop=True)
input_data.drop(columns=["round_x","round_y"],inplace=True)
input_data
#%%
xy_input=input_data.groupby(['x','y'],as_index=False).mean()[['x','y']]
boring_list=[]
for x,y in xy_input.values:
    boring=input_data[(input_data['x']==x) & (input_data['y']==y)].copy()
    boring['h_z']=boring['h']-boring['z']
    boring['h_z']=boring['h_z'].round(-1)
    boring=boring.groupby(['x','y','h_z'],as_index=False).mean()
    boring=boring[['x','y','h','z','h_z','t']]
    boring_list.append(boring)
input_data=pd.concat(boring_list).reset_index(drop=True)
# input_data.to_csv('./input/useful_data/input_data.csv',index=False)
# %%

#%%
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Yu Gothic' # font familyの設定
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['axes.titleweight']='bold'
plt.rcParams["font.size"] = 20

fig,ax=plt.subplots()
ax.grid()
ax.plot(input_data['t'],input_data['h_z'],'.')
ax.set_xlabel(u'地温(℃)')
ax.set_ylabel(u'標高(m)')
# %%
