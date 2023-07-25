#%%
import pandas as pd
import numpy as np
# from pandas.core.algorithms import diff
import matplotlib.pyplot as plt
import sys
# %%
name = "basic_curie_onsen_tishitsu_ohe_depth_grad"
data=pd.read_csv(f'./output_japan_last/voxler/nk/est_nk_output_{name}_detail.csv')
data
# %%
hyoukou = pd.read_csv("./input_japan/useful_data/est_grid_detail_ja.csv").groupby(["x","y"],as_index=False).mean()[["x","y","h"]]
hyoukou
#%%
data=data.merge(hyoukou,how="left",on=["x","y"])
data["z"]=data["h"]-data["h_z"]
data=data[~(data.z<0)].reset_index(drop=True)
data.describe()
#%%
depth=1500
generation_type = "バイナリー"
data=data[data.z<=depth].reset_index(drop=True)
if generation_type=="蒸気フラッシュ":
    data=data[data.t>=180].reset_index(drop=True)#蒸気フラッシュ
    # data=data[data.t>=150].reset_index(drop=True)#蒸気フラッシュ
elif generation_type=="バイナリー":
    data=data[(data.t>=120) & (data.t<=180)].reset_index(drop=True)#バイナリー
    # data=data[(data.t>=120) & (data.t<=150)].reset_index(drop=True)#バイナリー
elif generation_type=="低温バイナリー":
    data=data[(data.t>=80) & (data.t<=120)].reset_index(drop=True)#低温バイナリー
else:
    raise ValueError("error")
xy=data.groupby(["x","y"],as_index=False).max()[["x","y"]]
# %%
df_list=[]
for x,y in xy.values:
    pc=data[(data.x==x)&(data.y==y)]
    df = pd.DataFrame(np.zeros((1,5)),columns=["x","y","diff_z","mean_z","mean_t"])
    df.x,df.y=x,y
    df.diff_z=pc.z.max()-pc.z.min()
    df.mean_z = (pc.z.max()+pc.z.min())/2
    df.mean_t=pc.t.mean()
    # print(pc.z.max()-pc.z.min())
    # plt.plot(pc.t,-pc.z,"o")
    # plt.show()
    df_list.append(df)
df_list=pd.concat(df_list)
df_list=df_list[df_list.diff_z>0].reset_index(drop=True)
df_list
# %%
df_list.describe()
#%%
steam_table = pd.read_csv("./input_japan/usgs/steam_table.csv")
# steam_table.loc[steam_table.t==np.array([10,20]),"toro"] 
tal_dict = dict([(int(i),int(j)) for i,j in zip(steam_table.t,steam_table.tal)])
toro_dict = dict([(int(i),float(j)) for i,j in zip(steam_table.t,steam_table.toro)])
taltoro = np.polyfit(steam_table.tal,steam_table.toro,2)

#%%
plt.plot(steam_table.t,steam_table.tal)

#%%
df_list.mean_z

#%%
df_list["tal"] = df_list.mean_t.round(-1).apply(lambda x:tal_dict[x])
df_list["toro"] = df_list.mean_t.round(-1).apply(lambda x:toro_dict[x])
df_list
#%%
#%%
df_list["qr"]=2.7*(10**(6))*1000*1000*df_list.diff_z*(df_list.mean_t-15)
df_list["qwh"] = 0.25*df_list.qr/1000 #kJ
df_list["hwh"] = df_list.tal - (df_list.mean_z*9.8)/1000
df_list["Wa"]= (df_list.qwh/(df_list.hwh-tal_dict[15]))*(df_list.hwh-tal_dict[15]-(273.14+15)*(np.poly1d(taltoro)(df_list.hwh)-toro_dict[15]))
df_list["E"] =0.4*df_list.Wa/(30*365*24*60*60) #kW/km2
# df_list["E"] =df_list
df_list.describe()
#%%
df_list.E.sum()*25

#%%
df_list["h_z"]=0
df_list=df_list[["x","y","h_z","E"]]
# %%
df_list.to_csv(f"./output_japan_last/usgs/select/{generation_type}{depth}.csv",index=False)
# %%
df_list.describe()
# %%
df_list.hwh-tal_dict[15]-(273.14+15)*(np.poly1d(taltoro)(df_list.hwh)-toro_dict[15])
#%%
np.poly1d(taltoro)(df_list.hwh)
#%%
name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
data=pd.read_csv(f'./output_japan_last/voxler/nk/vox_est_nk_output_{name}_detail.csv')
hyoukou = pd.read_csv("./input_japan/useful_data/est_grid_detail_ja.csv").groupby(["x","y"],as_index=False).mean()[["x","y","h"]]
data
#%%
data=data.merge(hyoukou,how="left",on=["x","y"])
data.h=data.h.fillna(0)
data
#%%
data["z"]=data["h"]-data["h_z"]
# %%
data["q"]=0
#%%
data.loc[(data.z<=2000)&(data.t>=150),"q"]=3
data.loc[(data.z<=2000)&(data.t>=120) & (data.t<=150),"q"]=2
data.loc[(data.z<=2000)&(data.t>=80) & (data.t<=120),"q"]=1
data

# %%
# data.to_csv(f"./output_japan_last/usgs/kari.csv",index=False)
# %%
name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
data=pd.read_csv(f'./output_japan_last/voxler/nk/est_nk100_output_{name}_detail.csv')
# %%
data.describe()
# %%
data = pd.read_csv("./output_japan_last/usgs/蒸気フラッシュ1500.csv")
data.E.sum()*25
# %%
data.describe()
# %%
data.sort_values("E",ascending=False)[:20]
# %%
data=pd.read_csv(f'./input_japan/curie_point/add_grid_curie_detail_ja.csv')
data.describe()

# %%
