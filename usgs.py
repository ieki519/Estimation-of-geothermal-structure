#%%
import pandas as pd
import numpy as np
from pandas.core.algorithms import diff
# %%
data=pd.read_csv("./output/voxler/est_nk_inter100_detail.csv")
hyoukou=pd.read_csv("./input/elevation/hyoukou_albers_detail.csv")
hyoukou.drop("Unnamed: 0",axis=1,inplace=True)
hyoukou=hyoukou.rename(columns={"hyoukou":"h"})
data=data.merge(hyoukou,how="left",on=["x","y"])
data["z"]=data["h"]-data["h_z"]
data=data[~(data.z<0)].reset_index(drop=True)
data
# %%

data=data[data.z<=2000].reset_index(drop=True)
# data=data[data.t>=150].reset_index(drop=True)#蒸気フラッシュ
# data=data[(data.t>=120) & (data.t<=150)].reset_index(drop=True)#バイナリー
data=data[(data.t>=80) & (data.t<=120)].reset_index(drop=True)#低音バイナリー
xy=data.groupby(["x","y"],as_index=False).max()[["x","y"]]
# %%
df_list=[]
for x,y in xy.values:
    pc=data[(data.x==x)&(data.y==y)]
    df = pd.DataFrame(np.zeros((1,4)),columns=["x","y","diff_z","mean_t"])
    df.x,df.y=x,y
    df.diff_z=pc.z.max()-pc.z.min()
    df.mean_t=pc.t.mean()
    df_list.append(df)
df_list=pd.concat(df_list)
df_list=df_list[df_list.diff_z>0].reset_index(drop=True)
df_list
# %%
df_list["kJ"]=2.7*(10**(-6))*5000*5000*df_list.diff_z*(df_list.mean_t-80)/1000
df_list["h_z"]=0
df_list=df_list[["x","y","h_z","kJ"]]
df_list
# %%
# df_list.to_csv("./output/usgs/低音バイナリー80_120.csv",index=False)
# %%
df_list.describe()
# %%
