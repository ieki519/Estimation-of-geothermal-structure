#%%
import pandas as pd
# %%
# name = "basic_curie_onsen_depth_grad"
# name = "basic_volcano_curie_onsen_tishitsu_rank_depth_grad"#"basic_volcano_curie_onsen_depth_grad"#"basic_onsen_tishitsu_ohe_depth_grad"
# name = "basic_curie_onsen_tishitsu_rank_depth_grad"
# name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
name = "basic_curie_onsen_tishitsu_ohe_depth_grad"
# name="basic"

# est_500 = pd.read_csv(f'./output_japan/voxler/nk/est_nk_output_{name}.csv')
est_detail = pd.read_csv(f'./output_japan_last/voxler/nk/est_nk_output_{name}_detail.csv') 
est_detail
# %%
def grid_albers(sep_xy,sep_z):#sep_xy(km),sep_z(m)#単位に注意
    #sep_xy(km),sep_z(m)#単位に注意
    #df1
    ido=range(-200,1200,sep_xy)
    keido=range(-2100,-300,sep_xy)
    z=range(1500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df=pd.DataFrame(data,columns=['y','x','h_z'])

    df=df.loc[:,['x','y','h_z']]
    df=df.drop_duplicates(['x','y','h_z'])
    
    return df
# est_500_0 = grid_albers(10,500)
est_detail_0= grid_albers(5,100)
# est_500_0["t"]=0
est_detail_0["t"]=0
# %%
# est_500=pd.concat([est_500,est_500_0]).drop_duplicates(subset=["x","y","h_z"])
# est_500.to_csv(f'./output_japan/voxler/nk/vox_est_nk_output_{name}.csv',index = False)
est_detail=pd.concat([est_detail,est_detail_0]).drop_duplicates(subset=["x","y","h_z"])
est_detail.to_csv(f'./output_japan_last/voxler/nk/vox_est_nk_output_{name}_detail.csv',index = False)
# %%
name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
a=pd.read_csv(f'./output_japan_last/voxler/nk/vox_est_nk500_output_{name}_detail.csv')
a[a.t==a.t.max()]
# %%
hyoukou = pd.read_csv("./input_japan/useful_data/est_grid_detail_ja.csv").groupby(["x","y"],as_index=False).mean()[["x","y","h"]]
hyoukou
#%%
rinkai = a[a.t>=374].groupby(["x","y"],as_index=False).max().merge(hyoukou,on=["x","y"],how="left")
rinkai
#%%
rinkai["z"]=rinkai.h-rinkai.h_z
rinkai

#%%
# rinkai.x ,rinkai.y=rinkai.x.round(-4),rinkai.y.round(-4)
rinkai
#%%
rinkai.groupby(["x","y"],as_index=False).min().sort_values("z")[:50]#

#%%
name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
a=pd.read_csv(f'./output_japan_last/voxler/nk/vox_est_nk500_output_{name}_detail.csv').groupby(["x","y"],as_index=False).mean()
a.h_z=175000
a=a[["x","y","h_z","t"]]
a=a[a.y<((7/9)*a.x+(4300000/3))]
a.to_csv("./a.csv",index=False)
a
#%%
a[a.h_z==-3000].to_csv(f'./output_japan_last/voxler/nk/vox_est_dnn_output_{name}_detail_3000.csv',index = False)
# %%
name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
data=pd.read_csv(f'./output_japan_last/voxler/nk/est_nk100_output_{name}_detail.csv')
hyoukou = pd.read_csv("./input_japan/useful_data/est_grid_detail_ja.csv").groupby(["x","y"],as_index=False).mean()[["x","y","h"]]
data=data.merge(hyoukou,on=["x","y"],how="left")
data["z"]=data["h"]-data["h_z"]
data=data[data.z>0]
data
#%%
data[(data.x==-810000)&(data.y==565000)&(data.h_z==-3000)]
# %%
data[(data.x==-815000)&(data.y==565000)&(data.t>=374)].min()
#%%
import seaborn as sns
sns.set()
fig,ax=plt.subplots(figsize=(5,10))
kakkonda = data[(data.x==-810000)&(data.y==565000)][::3]
kakkonda=kakkonda[kakkonda.z<=4000]
ax.plot(kakkonda.t,-kakkonda.z,)
#%%
a[a.h_z==-3700].sort_values("t",ascending=False)[:20]
# %%
curie = pd.read_csv("./input_japan/curie_point/add_grid_curie_detail_ja.csv")
curie
# %%
curie[(curie.x==-815000)&(curie.y==565000)]
# %%
a = pd.read_csv("./input_japan/useful_data/input_data_ja.csv")#.groupby(["x","y"],as_index=False).max()
a = a[a.t>0]
# %%
# plt.hist(a.t,bins=100)
plt.hist(np.log(a.t),bins=100)
# %%
print(data)
plt.hist(a.t, bins=np.logspace(0, 100000, 500))
plt.gca().set_xscale("log")
# %%
a=pd.read_csv("./input_japan/useful_data/est_grid_detail_ja.csv")
a.h=a.h*100+175000
a.groupby(["x","y"],as_index=False).mean().to_csv("./output_japan_last/voxler/nk/est_grid_h.csv",index=False)
# %%
from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np
# %%
cdist(np.array())