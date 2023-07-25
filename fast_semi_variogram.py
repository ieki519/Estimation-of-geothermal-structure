#%%
import pandas as pd
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform
import time
# %%
data=pd.read_csv('./input/useful_data/est_grid_500.csv')
data['t']=np.random.rand(len(data))
data
#%%
xy=data.groupby(['x','y'],as_index=False).mean()[['x','y']]
yz=data.groupby(['y','h_z'],as_index=False).mean()[['y','h_z']]
zx=data.groupby(['h_z','x'],as_index=False).mean()[['h_z','x']]
# %%
xy_dist=squareform(pdist(xy))
yz_dist=squareform(pdist(yz))
zx_dist=squareform(pdist(zx)) 
# %%
near_list_xy=[]
for i in range(len(xy_dist)):
    near_list_xy.append(np.where(((xy_dist[i,:]<80000)&(0<xy_dist[i,:]))))

temp=data['t'].values
temp_d=np.zeros((len(temp),len(temp)))
start_time=time.time()
for j in range(len(temp)):
    print(j)
    temp_d[j][near_list_xy[j][0]]=np.sqrt((temp[j]-temp[near_list_xy[j][0]])**2)
print(time.time()-start_time)
# %%
near_list_xy[0][0]
# %%
temp[0]-temp[near_list_xy[0][0]]
# %%
len(temp_d)
# %%
