#%%
import pandas as pd
import numpy as np
# %%
profile_db=pd.read_csv('./input/useful_data/database/profile_db.csv')
geothermal_db=pd.read_csv('./input/useful_data/database/geothermal_db.csv')
GSJ_db=pd.read_csv('./input/useful_data/database/GSJ_db.csv')
# %%
input_data=pd.concat([profile_db,geothermal_db,GSJ_db]).reset_index(drop=True)
xy_input=input_data.groupby(['x','y'],as_index=False).mean()[['x','y']]

xy_input=xy_input.round()
profile_db['x'],profile_db['y']=profile_db['x'].round(),profile_db['y'].round()
geothermal_db['x'],geothermal_db['y']=geothermal_db['x'].round(),geothermal_db['y'].round()
GSJ_db['x'],GSJ_db['y']=GSJ_db['x'].round(),GSJ_db['y'].round()

xy_input=xy_input[xy_input.duplicated()]
for x,y in xy_input.values:
    GSJ_db=GSJ_db[~((GSJ_db['x']==x)&(GSJ_db['y']==y))]
GSJ_db.groupby(['x','y'],as_index=False).mean()[['x','y']]
#%%
input_data=pd.concat([profile_db,geothermal_db,GSJ_db]).reset_index(drop=True)
#%%
# xy_input=input_data.groupby(['x','y'],as_index=False).mean()[['x','y']]
# boring_list=[]
# for x,y in xy_input.values:
#     boring=input_data[(input_data['x']==x) & (input_data['y']==y)].copy()
#     boring['h_z']=boring['h']-boring['z']
#     boring['h_z']=boring['h_z'].round(-1)
#     boring=boring.groupby(['x','y','h_z'],as_index=False).mean()
#     boring=boring[['x','y','h','z','h_z','t']]
#     boring_list.append(boring)
# input_data=pd.concat(boring_list).reset_index(drop=True)
# input_data.to_csv('./input/useful_data/input_data.csv',index=False)
# %%
xy_input=input_data.groupby(['x','y'],as_index=False).mean()[['x','y']]
boring_list=[]
for x,y in xy_input.values:
    boring=input_data[(input_data['x']==x) & (input_data['y']==y)].copy()
    boring['z']=boring['z'].round(-1)
    boring=boring.groupby(['x','y','h','z'],as_index=False).mean()
    boring['h_z']=boring['h']-boring['z']
    boring=boring[['x','y','h','z','h_z','t']]
    boring_list.append(boring)
input_data=pd.concat(boring_list).reset_index(drop=True)
# input_data.to_csv('./input/useful_data/input_data.csv',index=False)
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
profile_db=pd.read_csv('./input/useful_data/database/profile_db.csv')
profile_db['x'],profile_db['y']=profile_db['x'].round(-2),profile_db['y'].round(-2)
profile_db_xy = profile_db.groupby(['x','y'],as_index=False).mean()[['x','y']]
profile_db_xy
# %%
