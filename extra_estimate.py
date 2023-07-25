#%%
from cv2 import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import glob
import re
from sklearn.model_selection import train_test_split
# %%
input_data=pd.read_csv("./input_japan/useful_data/input_data_ja.csv")
input_data
# %%
# name = "basic_volcano_curie_onsen_tishitsu_rank_depth800_grad"
name="basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
num=0
est_data = pd.read_csv(f'./output_japan_last/voxler/nk/est_nk500_output_{name}_detail_input_xy.csv')
est_data
# %%
xy = input_data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
xy.shape
# %%
num_list=[]
for i in glob.glob("c:/Users/Admin/Desktop/修士論文/vscode_ja/小池先生/*.png"):
    num_list.append(int(i.split(".")[0].split("\\")[1]))
num_list
#%%
for i in range(xy.shape[0]):
    print(i)
    number = 817
    x=xy.x[number]
    y=xy.y[number]
    train=input_data[(input_data.x==x)&(input_data.y==y)]
    
    # df = est_data[(est_data.x==x)&(est_data.y==y)]
    # abcd = np.polyfit(df.h_z.values,df.t.values,8)
    # est_data.loc[(est_data.x==x)&(est_data.y==y),"t"]=np.poly1d(abcd)(df.h_z.values)
    
    test = est_data[(est_data.x==x)&(est_data.y==y)][::30].reset_index(drop=True)
    min=train.h_z.min()
    test = test[test.h_z<min]
    fig, ax = plt.subplots()
    # ax.set_xlim(0,500)
    # ax.set_ylim(-2000,0)
    ax.set_xlabel('Temperature(℃)') 
    ax.set_ylabel('Elevation(m)')
    ax.grid()
    plt.rcParams["font.size"] = 12
    ax.plot(train["t"],train["h_z"],c="black")
    ax.plot(test["t"],test["h_z"],c="red")
    plt.show()
    break
    # fig.savefig(f"./小池先生/500/{i}_500.png",bbox_inches = 'tight')
    if i==300:
        break
# %%
# %%
xy.shape
#%%
def extra_split(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.1, shuffle=False)
    return train_data, test_data
#%%
input_data=pd.read_csv("./input_japan/useful_data/input_data_ja.csv")
_ , input_data = extra_split(input_data)
input_data
# %%
name = "basic_volcano_curie_onsen_tishitsu_rank_depth800_grad"
num=0
est_data = pd.read_csv(f'./output_japan/voxler/est_nk_extra_{name}_{num}_input_xy.csv')
est_data
xy_est = est_data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
for x,y in xy_est.values:
    df = est_data[(est_data.x==x)&(est_data.y==y)]
    abcd = np.polyfit(df.h_z.values,df.t.values,4)
    # plt.plot(,df.h_z.values)
    # plt.plot(df.t.values,df.h_z.values)
    est_data.loc[(est_data.x==x)&(est_data.y==y),"t"]=np.poly1d(abcd)(df.h_z.values)
    
    break
#%%
input_data.h_z = input_data.h_z.round(-1)
input_data
# %%
est_data
# %%
a = input_data.merge(est_data,how="left",on=["x","y","h_z"])
a=a.dropna()
# %%
np.sqrt(mean_squared_error(a.t_x.values,a.t_y.values))
# %%

# %%
