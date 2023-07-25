#%%
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform,cdist
#%%
#grid
#%%
volcano=pd.read_csv('./input_japan/volcano/volcano_ja.csv')
volcano=volcano.values
volcano
#%%
xy_grid_albers=pd.read_csv('./input_japan/useful_data/est_grid_detail_ja.csv')
xy_grid_albers=xy_grid_albers.groupby(['x','y'],as_index=False).mean()[['x','y']].values
xy_grid_albers.shape
# %%
volcano_dist=cdist(xy_grid_albers,volcano)
# %%
volcano_dist
# %%
volcano_dist_list=[]
for i in range(volcano_dist.shape[0]):
    volcano_dist_list.append(np.min(volcano_dist[i,:]))
volcano_dist_list
# %%
xy_grid_albers_master=pd.read_csv('./input_japan/useful_data/est_grid_detail_ja.csv')
xy_grid_albers_master=xy_grid_albers_master.groupby(['x','y'],as_index=False).mean()[['x','y']]
xy_grid_albers_master['volcano']=volcano_dist_list
xy_grid_albers_master
# %%
xy_grid_albers_master.to_csv('./input_japan/volcano/add_grid_volcano_ja.csv',index=False)
# %%
#input
#%%
volcano=pd.read_csv('./input_japan/volcano/volcano_ja.csv')
volcano=volcano.values
volcano

input_data=pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data=input_data.groupby(['x','y'],as_index=False).mean()[['x','y']].values
input_data.shape

volcano_dist=cdist(input_data,volcano)
volcano_dist

volcano_dist_list=[]
for i in range(volcano_dist.shape[0]):
    volcano_dist_list.append(np.min(volcano_dist[i,:]))
volcano_dist_list

input_data_master=pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data_master=input_data_master.groupby(['x','y'],as_index=False).mean()[['x','y']]
input_data_master['volcano']=volcano_dist_list
input_data_master
#%%
# input_data_master.to_csv('./input_japan/volcano/add_input_volcano_ja.csv',index=False)
# %%
#%%
#onsen
volcano=pd.read_csv('./input_japan/volcano/volcano_ja.csv')
volcano=volcano.values
volcano

onsen = pd.read_csv('./input_japan/onsen/database/onsen_xyh_ja.csv')
onsen=onsen.groupby(['x','y'],as_index=False).mean()[['x','y']].values
onsen.shape

volcano_dist=cdist(onsen,volcano)
volcano_dist

volcano_dist_list=[]
for i in range(volcano_dist.shape[0]):
    volcano_dist_list.append(np.min(volcano_dist[i,:]))
volcano_dist_list

onsen_master = pd.read_csv('./input_japan/onsen/database/onsen_xyh_ja.csv')
onsen_master=onsen_master.groupby(['x','y'],as_index=False).mean()[['x','y']]
onsen_master['volcano']=volcano_dist_list
onsen_master.to_csv('./input_japan/onsen/add_onsen_volcano_ja.csv',index=False)
#%%