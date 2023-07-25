#%%
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform,cdist
import pykrige.variogram_models as vm
from tqdm import tqdm
#%%
#grid_data
#%%
grid_data_list=[]
for key in tqdm(['hokkaidou','higashi','nishi','kyusyu']):
    grid_data=pd.read_csv('./input/useful_data/est_grid_500.csv')
    # grid_xy=grid_data.groupby(['x','y'],as_index=False).mean()[['x','y']].values
    input_data=pd.read_csv('./input/useful_data/input_data.csv')
    # input_xy=input_data.groupby(['x','y'],as_index=False).mean()[['x','y']].values
    if key=='hokkaidou':
        #grid
        grid_data=grid_data[(grid_data['x']-grid_data['y'])<=1200000]
        grid_data['h_z*']=grid_data['h_z']*(40/1.5)
        #input
        input_data=input_data[(input_data['x']-input_data['y'])<=1200000]
        input_data['h_z*']=input_data['h_z']*(40/1.5)
    elif key=='higashi':
        #grid
        grid_data=grid_data[((grid_data['x']-grid_data['y'])>1200000)&((grid_data['x']+grid_data['y'])>=5000000)]
        grid_data['h_z*']=grid_data['h_z']*(80/1.5)
        #input
        input_data=input_data[((input_data['x']-input_data['y'])>1200000)&((input_data['x']+input_data['y'])>=5000000)]
        input_data['h_z*']=input_data['h_z']*(80/1.5)   
    elif key=='nishi':
        #grid
        grid_data=grid_data[((grid_data['y']+(2/3)*grid_data['x'])>3050000)&((grid_data['x']+grid_data['y'])<5000000)]
        grid_data['h_z*']=grid_data['h_z']*(80/1.5)
        #input
        input_data=input_data[((input_data['y']+(2/3)*input_data['x'])>3050000)&((input_data['x']+input_data['y'])<5000000)]
        input_data['h_z*']=input_data['h_z']*(80/1.5)   
    elif key=='kyusyu':
        #grid
        grid_data=grid_data[(grid_data['y']+(2/3)*grid_data['x'])<=3050000]
        grid_data['h_z*']=grid_data['h_z']*(80/1.5)
        #input
        input_data=input_data[(input_data['y']+(2/3)*input_data['x'])<=3050000]
        input_data['h_z*']=input_data['h_z']*(80/1.5)   

    grid_xyz=grid_data[['x','y','h_z*']].values
    input_xyz=input_data[['x','y','h_z*']].values

    pair_dist_xyz=cdist(grid_xyz,input_xyz)

    t_list=[]
    d_list=[]
    pair_dist_xyz=np.where(pair_dist_xyz==0,np.inf,pair_dist_xyz)
    for i in range(pair_dist_xyz.shape[0]):
        idx_min=np.argmin(pair_dist_xyz[i,:])
        t_list.append(input_data['t'].values[idx_min])
        d_list.append(pair_dist_xyz[i,idx_min])

    grid_data['near_t']=t_list
    grid_data['near_dist']=d_list
    if key=='hokkaidou':
        grid_data['variogram']=vm.spherical_variogram_model([2500-70,25000,70],grid_data['near_dist'].values)
    elif key=='higashi':
        grid_data['variogram']=vm.spherical_variogram_model([8000-550,45000,550],grid_data['near_dist'].values)
    elif key=='nishi':
        grid_data['variogram']=vm.spherical_variogram_model([1500-80,3000,80],grid_data['near_dist'].values)
    elif key=='kyusyu':
        grid_data['variogram']=vm.spherical_variogram_model([4000-250,50000,250],grid_data['near_dist'].values)
    grid_data_list.append(grid_data)
# %%
grid_data_list=pd.concat(grid_data_list)
grid_data_list
# %%
grid_data_master=pd.read_csv('./input/useful_data/est_grid_500.csv')
grid_data_master
# %%
grid_data_master=grid_data_master.merge(grid_data_list,how='left',on=['x','y','h','z','h_z'])
grid_data_master
# %%
grid_data_master=grid_data_master[['x','y','h','z','h_z','near_t','near_dist','variogram']]
grid_data_master
# %%
# grid_data_master.to_csv('./input/grid_albers_xyhx_v.csv',index=False)
# %%
# input_data
# %%
