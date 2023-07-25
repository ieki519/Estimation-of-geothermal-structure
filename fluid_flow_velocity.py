#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# %% 
# detail 
data =pd.read_csv("./output/voxler/est_nk_inter_addz_detail.csv")
hyoukou =pd.read_csv("./input/elevation/hyoukou_albers_detail.csv")[["x","y","hyoukou"]]
hyoukou=hyoukou.rename(columns={"hyoukou":"h"})
data= data.merge(hyoukou,how="left",on=["x","y"])
data["z"]=data.h-data.h_z
data = data[data["z"]>0].reset_index(drop=True)
data["v"] = (2.2/(1000*4.185*(10**3)*data.z))*(np.log(800-0)-np.log(800-data.t))
data = data[["x","y","h_z","v"]]
data.describe()
# %%
xy_unique = data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
yz_unique = data.groupby(["y","h_z"],as_index=False).mean()[["y","h_z"]]
zx_unique = data.groupby(["h_z","x"],as_index=False).mean()[["h_z","x"]]
zx_unique
# %%
for x,y in tqdm(xy_unique.values):
    pc = data[(data.x==x)&(data.y==y)]
    for h_z in pc.h_z.values:
        v_center = pc[pc.h_z==h_z].v.values
        v_neg = pc[pc.h_z==h_z-100].v.values
        v_plus = pc[pc.h_z==h_z+100].v.values
        if v_center.size and v_neg.size and v_plus.size :
            v_k =np.polyfit([-100,0,100],[v_neg[0],v_center[0],v_plus[0]],1)[0]
            data.loc[(data.x==x)&(data.y==y)&(data.h_z==h_z),"v_k"] = v_k

for y,h_z in tqdm(yz_unique.values):
    pc = data[(data.y==y)&(data.h_z==h_z)]
    for x in pc.x.values:
        v_center = pc[pc.x==x].v.values
        v_neg = pc[pc.x==x-5000].v.values
        v_plus = pc[pc.x==x+5000].v.values
        if v_center.size and v_neg.size and v_plus.size :
            v_i =np.polyfit([-5000,0,5000],[v_neg[0],v_center[0],v_plus[0]],1)[0]
            data.loc[(data.x==x)&(data.y==y)&(data.h_z==h_z),"v_i"] = v_i
            
for z,x in tqdm(zx_unique.values):
    pc = data[(data.z==z)&(data.x==x)]
    for y in pc.y.values:
        v_center = pc[pc.y==y].v.values
        v_neg = pc[pc.y==y-5000].v.values
        v_plus = pc[pc.y==y+5000].v.values
        if v_center.size and v_neg.size and v_plus.size :
            v_j =np.polyfit([-5000,0,5000],[v_neg[0],v_center[0],v_plus[0]],1)[0]
            data.loc[(data.x==x)&(data.y==y)&(data.h_z==h_z),"v_j"] = v_j
data
# %%
# not detail 
data =pd.read_csv("./output/voxler/est_nk_inter_addz_new.csv")
hyoukou =pd.read_csv("./input/elevation/hyoukou_albers.csv")[["x","y","hyoukou"]]
hyoukou=hyoukou.rename(columns={"hyoukou":"h"})
data= data.merge(hyoukou,how="left",on=["x","y"])
data["z"]=data.h-data.h_z
data = data[data["z"]>0].reset_index(drop=True)
data["v"] = (2.2/(1000*4.185*(10**3)*data.z))*(np.log(800-data.t)-np.log(800-0))
data = data[["x","y","h_z","v"]]
data.to_csv("./output/fluid_flow_velocity/est_nk_inter_addz_new_v.csv")
# %%
data.describe()
#%%
xy_unique = data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
yz_unique = data.groupby(["y","h_z"],as_index=False).mean()[["y","h_z"]]
zx_unique = data.groupby(["h_z","x"],as_index=False).mean()[["h_z","x"]]
zx_unique
# %%
for x,y in tqdm(xy_unique.values):
    pc = data[(data.x==x)&(data.y==y)]
    for h_z in pc.h_z.values:
        v_center = pc[pc.h_z==h_z].v.values
        v_neg = pc[pc.h_z==h_z-500].v.values
        v_plus = pc[pc.h_z==h_z+500].v.values
        if v_center.size and v_neg.size and v_plus.size :
            v_k =np.polyfit([-500,0,500],[v_neg[0],v_center[0],v_plus[0]],1)[0]
            data.loc[(data.x==x)&(data.y==y)&(data.h_z==h_z),"v_k"] = v_k

for y,h_z in tqdm(yz_unique.values):
    pc = data[(data.y==y)&(data.h_z==h_z)]
    for x in pc.x.values:
        v_center = pc[pc.x==x].v.values
        v_neg = pc[pc.x==x-10000].v.values
        v_plus = pc[pc.x==x+10000].v.values
        if v_center.size and v_neg.size and v_plus.size :
            v_i =np.polyfit([-10000,0,10000],[v_neg[0],v_center[0],v_plus[0]],1)[0]
            data.loc[(data.x==x)&(data.y==y)&(data.h_z==h_z),"v_i"] = v_i

for h_z,x in tqdm(zx_unique.values):
    pc = data[(data.h_z==h_z)&(data.x==x)]
    for y in pc.y.values:
        v_center = pc[pc.y==y].v.values
        v_neg = pc[pc.y==y-10000].v.values
        v_plus = pc[pc.y==y+10000].v.values
        if v_center.size and v_neg.size and v_plus.size :
            v_j =np.polyfit([-10000,0,10000],[v_neg[0],v_center[0],v_plus[0]],1)[0]
            data.loc[(data.x==x)&(data.y==y)&(data.h_z==h_z),"v_j"] = v_j
data=data.dropna()
data=data[["x","y","h_z","v_i","v_j","v_k"]]
data = data.reset_index(drop=True)
data
# %%
data.to_csv("./output/fluid_flow_velocity/est_nk_inter_addz_new_velocity.csv",index=False)
# %%
data.describe()
# %%
data = pd.read_csv("./output/fluid_flow_velocity/est_nk_inter_addz_new_velocity.csv")
data["v_i"]=data["v_i"]*10**15
data["v_j"]=data["v_j"]*10**15
data["v_k"]=data["v_k"]*10**15
data.describe()
# %%
# data[data.y==2*10**6]
# #%%
data[data.y==2*10**6].to_csv("./output/fluid_flow_velocity/est_nk_inter_addz_new_velocity_norm1000_yz.csv",index=False)
#%%
data.to_csv("./output/fluid_flow_velocity/est_nk_inter_addz_new_velocity_norm.csv",index=False)
# %%
data = pd.read_csv("./output/fluid_flow_velocity/est_nk_inter_addz_velocity.csv")
data
# %%
plt.plot(data.v_i)
# %%
