#%%
import pandas as pd 
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
#%%
data=pd.read_csv('./input/useful_data/input_data.csv')
features=['x','y','h_z','t']
data=data[features]
data['h_z_round']=data['h_z'].round(-2)
z_unique=data['h_z_round'].unique()
z_unique
# %%
pc_list=[]
for z in tqdm(z_unique):
    pc=data[data['h_z_round']==z].reset_index(drop=True)
    if len(pc)==1:
        continue
    else:
        xy_dist=squareform(pdist(pc[['x','y']]))
        near_t=[]
        near_dist=[]
        for i in range(len(pc)):
            xy_dist[i,:]=np.where(xy_dist[i,:]==0,np.inf,xy_dist[i,:])
            near_idx=np.argmin(xy_dist[i,:])
            near_t.append(pc['t'][near_idx])
            near_dist.append(xy_dist[i,near_idx])
        pc['near_t']=near_t
        pc['near_dist']=near_dist
    pc_list.append(pc)
pc=pd.concat(pc_list)
# %%
data=data.merge(pc,how='left',on=['x','y','h_z','h_z_round','t'])
data
# %%
pc
# %%
data['near_t']-data['t']
# %%
