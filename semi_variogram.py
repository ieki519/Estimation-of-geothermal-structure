#%%
import enum
from numpy.core.fromnumeric import argmax
import pandas as pd 
import numpy as np
import math
from scipy.spatial.distance import pdist, squareform,cdist
import matplotlib.pyplot as plt
from tqdm import tqdm
import pykrige.variogram_models as vm
import time
#%%
data=pd.read_csv('./input/useful_data/input_data.csv')
# %%
features=['x','y','h_z','t']
data_val=data[features]
data_val
# %%
data_val=data_val[(data_val['x']-data_val['y'])<=1200000]
data_val_z=data_val.copy()
data_val_z['h_z']=data_val_z['h_z'].round(-2)
data_val_z=data_val_z.groupby(['x','y','h_z'],as_index=False).mean()
data_val_z
# %%
z_unique=data_val_z['h_z'].unique()
z_unique
# %%
sv_all=[]
for z in z_unique:
    pc=data_val_z[data_val_z['h_z']==z]
    pc_xy_dis=squareform(pdist(pc[['x','y']]))
    pc_t_vario=squareform((pdist(pc['t'].values.reshape(-1,1))**2)/2)
    sep=11000
    sv_i=np.zeros(len(range(0,200001,sep)))
    for i,value in enumerate(range(0,200001,sep)):
        mask1=pc_xy_dis>value
        mask2=pc_xy_dis<value+sep
        mask=mask1*mask2
        res1=pc_t_vario[mask]
        mask3=res1>0
        res2=(res1[mask3].mean())
        sv_i[i]=res2
    sv_all.append(sv_i)
sv_all=np.vstack(sv_all)
result=[]
for i in range(len(range(0,200001,sep))):
    result.append(sv_all[:,i][~np.isnan(sv_all[:,i])].mean())
# %%
plt.plot(range(0,200001,sep),result)
# %%

# 全球

# %%
data=pd.read_csv('./input/useful_data/input_data.csv')
features=['x','y','h_z','t']
data=data[features]
keys=['hokkaidou','higashi','nishi','kyusyu']
key=keys[3]
if key=='hokkaidou':
    data=data[(data['x']-data['y'])<=1200000]
    data['h_z']=data['h_z']*(40/1.5)
elif key=='higashi':
    data=data[((data['x']-data['y'])>1200000)&((data['x']+data['y'])>=5000000)]
    data['h_z']=data['h_z']*(80/1.5)
elif key=='nishi':
    data=data[((data['y']+(2/3)*data['x'])>3050000)&((data['x']+data['y'])<5000000)]
    data['h_z']=data['h_z']*(80/1.5)
elif key=='kyusyu':
    data=data[(data['y']+(2/3)*data['x'])<=3050000]
    data['h_z']=data['h_z']*(80/1.5)
data
# %%
data[['x','y','h_z']].astype('int64')
#%%
xy_dis=squareform(pdist(data[['x','y','h_z']].astype('int32')))
#%%
t_vario=squareform(pdist(data['t'].values.reshape(-1,1))**2)
# %%
sep=1100
max_dist=200001
sv_i=np.zeros(len(range(0,max_dist,sep)))
for i,value in enumerate(tqdm(range(0,max_dist,sep))):
    mask1=xy_dis>value
    mask2=xy_dis<value+sep
    mask=mask1*mask2
    res1=t_vario[mask]
    mask3=res1>0
    res2=(res1[mask3].mean())/2
    sv_i[i]=res2
sv_i
# %%
x=range(0,max_dist,sep)
plt.plot(x[:],sv_i[:],c='black',marker='o')
plt.plot(x[:],vm.spherical_variogram_model([4000-250,50000,250],np.array(x))[:],c='red')

#%%

# 高速(t-t_)**2
# %%
near_list=[]
for i in range(len(xy_dis)):
    near_list.append(np.where(((xy_dis[i,:]<25000)&(0<xy_dis[i,:]))))

temp=data['t'].values
temp_d=np.zeros((len(temp),len(temp)))
start_time=time.time()
for j in range(len(temp)):
    temp_d[j][near_list[j][0]]=np.sqrt((temp[j]-temp[near_list[j][0]])**2)
print(time.time()-start_time)
#%%
temp_d.shape
#%%
start_time=time.time()
t_vario=squareform(pdist(data['t'].values.reshape(-1,1))**2)
print(time.time()-start_time)
# %%
t_vario.shape
# %%
