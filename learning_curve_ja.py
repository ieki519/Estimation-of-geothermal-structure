#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
# hyper unit inter 
fig, ax = plt.subplots()
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,45)
ax.set_title("内挿評価")
# ax.set_ylim(40,60)
for i in [30,60,90,120,150,180]:
    lc=pd.read_csv(f'./output_japan/learning_curve/unit_hyper/lc_dnn_inter_basic_unit{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([30,60,90,120,150,180],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
fig.savefig(f"./修論用画像/hyper_unit_inter.png",dpi=300,bbox_inches='tight')
plt.show()
#%%
lc
#%%
# hyper unit extra
fig, ax = plt.subplots()
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_title("外挿評価")
ax.set_ylim(35,60)
for i in [30,60,90,120,150,180]:
    lc=pd.read_csv(f'./output_japan/learning_curve/unit_hyper/lc_dnn_extra_basic_unit{i}_0.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([30,60,90,120,150,180],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
fig.savefig(f"./修論用画像/hyper_unit_extra.png",dpi=300,bbox_inches='tight')
plt.show()
#%%
# hyper unit inter + extra
fig, ax = plt.subplots()
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_title("内挿評価と外挿評価の平均")

ax.set_ylim(35,45)
for i in [30,60,90,120,150,180]:
    lc_i=pd.read_csv(f'./output_japan/learning_curve/unit_hyper/lc_dnn_inter_basic_unit{i}.csv')#[:21]
    lc_e=pd.read_csv(f'./output_japan/learning_curve/unit_hyper/lc_dnn_extra_basic_unit{i}_0.csv')#[:21]
    # lc_t = pd.concat([lc_i["test_loss"],lc_e["test_loss"]],axis=1).mean(axis=1)
    # lc_t = ((1/5)*lc_i["test_loss"]+(1/10)*lc_e["test_loss"])/((1/5)+(1/10))
    lc_t = (lc_i["test_loss"]+lc_e["test_loss"])/2
    
    ax.plot(lc_i['epoch'],lc_t)
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([30,60,90,120,150,180],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
fig.savefig(f"./修論用画像/hyper_unit_interextra.png",dpi=300,bbox_inches='tight')
plt.show()
# ----------------------------------------------------------------------------------------------------
#%%
# hyper curie inter 
fig, ax = plt.subplots()
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,45)
ax.set_title("内挿評価")

# ax.set_ylim(30,40)
for i in [0,1,10,100,1000,10000,100000,1000000]:
    lc=pd.read_csv(f'./output_japan/learning_curve/curie_hyper/lc_dnn_inter_basic_curie{i}.csv')#[:16]
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([0,1,"1/10","1/100","1/1000","1/10000","1/100000","1/1000000"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
fig.savefig(f"./修論用画像/hyper_curie_inter.png",dpi=300,bbox_inches='tight')

#%%
# hyper curie extra
fig, ax = plt.subplots()
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_title("外挿評価")

# ax.set_ylim(35,60)
for i in [0,1,10,100,1000,10000,100000,1000000]:
    lc=pd.read_csv(f'./output_japan/learning_curve/curie_hyper/lc_dnn_extra_basic_curie{i}_0.csv')#[:16]
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
# ax.legend([0,1,10,100,1000,10000,100000,1000000],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
ax.legend([0,1,"1/10","1/100","1/1000","1/10000","1/100000","1/1000000"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.show()
fig.savefig(f"./修論用画像/hyper_curie_extra.png",dpi=300,bbox_inches='tight')

# %%
#%%
# hyper curie inter + extra
fig, ax = plt.subplots()
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_title("内挿評価と外挿評価の平均")

ax.set_ylim(35,50)
for i in [0,1,10,100,1000,10000,100000,1000000]:
    lc_i=pd.read_csv(f'./output_japan_last/learning_curve/curie_hyper/lc_dnn_inter_basic_curie{i}.csv')#[:11]
    lc_e=pd.read_csv(f'./output_japan_last/learning_curve/curie_hyper/lc_dnn_extra_basic_curie{i}_0.csv')#[:11]
    # lc_t = pd.concat([lc_i["test_loss"],lc_e["test_loss"]],axis=1).mean(axis=1)
    # lc_t = ((1/5)*lc_i["test_loss"]+(1/10)*lc_e["test_loss"])/((1/5)+(1/10))
    lc_t = (lc_i["test_loss"]+lc_e["test_loss"])/2
    ax.plot(lc_i['epoch'],lc_t)
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
# ax.legend([0,1,10,100,1000,10000,100000,1000000],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
ax.legend([0,1,"1/10","1/100","1/1000","1/10000","1/100000","1/1000000"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.show()
fig.savefig(f"./修論用画像/hyper_curie_interextra.png",dpi=300,bbox_inches='tight')

# %%
# ----------------------------------------------------------------------------------------------------
all_features_name_list=[]
features_name_list = ["volcano","curie","onsen","tishitsu","depth","grad"]
N=len(features_name_list)
for i in range(2**N):
    A=[]
    for j in range(N):
        if ((i>>j)&1):
            A.append(features_name_list[j])
    all_features_name_list.append(A)
all_features_name_list
all_name_list = []
for all_f in all_features_name_list:
    name ="basic"
    for f in all_f:
        name+="_"+f
    all_name_list.append(name)
all_name_list
#%%
# inter
lc_inter_dict = {}
for i in all_name_list:
    # if "curie" in i:
    #     continue
    lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_{i}.csv')
    lc_inter_dict[i] = lc["test_loss"].min()#.min()#[10]#.min()#5]
    # lc_list=[]
    # hyper_list =[0.1915366765502127,
    # 0.19733788835890165,
    # 0.20986420867173736,
    # 0.19222422757939064,
    # 0.20903699883975763]
    # for j in range(5):
    #     lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_{i}_{j}.csv')['test_loss'].values
    #     lc_list.append(hyper_list[j]*lc)
    # lc_i = np.vstack(lc_list).sum(axis=0)
    # lc_inter_dict[i] = lc_i.min()
lc_inter_dict=sorted(lc_inter_dict.items(),key= lambda x:x[1])[:10]
lc_inter_dict
#%%
# extra
lc_extra_dict = {}
for i in all_name_list:
    # if "curie" in i:
    #     continue
    if "depth" in i:
        i = i.split("depth")
        i.insert(1,"depth800")
        i = "".join(i)
    lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_extra_{i}_0.csv')
    lc_extra_dict[i] = lc["test_loss"].min()#.min()#[10]
lc_extra_dict=sorted(lc_extra_dict.items(),key= lambda x:x[1])[:10]
lc_extra_dict
# %%
# inter+extra
lc_inter_extra_dict = {}
for i in all_name_list:
    # if "volcano" in i:
    #     continue
    lc_i=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_{i}.csv')
    if "depth" in i:
        i = i.split("depth")
        i.insert(1,"depth800")
        i = "".join(i)
    lc_e=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_extra_{i}_0.csv')
    # idx = lc_i["test_loss"].idxmin()
    # lc_inter_extra_dict[i] = (idx,((1/5)*lc_i["test_loss"][idx]+(1/10)*lc_e["test_loss"][idx])/((1/5)+(1/10)))
    # lc_inter_extra_dict[i] = (lc_i["test_loss"][idx]+lc_e["test_loss"][idx])/2
    lc = ((1/5)*lc_i["test_loss"]+(1/10)*lc_e["test_loss"])/((1/5)+(1/10))
    lc_inter_extra_dict[i] = (lc.idxmin(),lc.min())
    # lc_inter_extra_dict[i] = (lc_i["test_loss"][10]+lc_e["test_loss"][10])/2
    # lc_inter_extra_dict[i] = (lc_i["test_loss"]+lc_e["test_loss"]).min()/2
    # lc_inter_extra_dict[i] = (idx,(lc_i["test_loss"].min()+lc_e["test_loss"].min())/2)
    # lc_inter_extra_dict[i] = (idx,(((1/5)*lc_i["test_loss"].min()+(1/10)*lc_e["test_loss"].min())/((1/5)+(1/10))))
lc_inter_extra_dict=sorted(lc_inter_extra_dict.items(),key= lambda x:x[1][1])[:5]
lc_inter_extra_dict
# %%
# ----------------------------------------------------------------------------------------------------
# inter+extra
lc_inter_extra_dict = {}
for i in ['basic_curie_onsen_depth_grad','basic_curie_onsen_tishitsu_depth_grad']:
    # if "volcano" in i:
    #     continue
    lc_i=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_{i}.csv')
    if "depth" in i:
        i = i.split("depth")
        i.insert(1,"depth800")
        i = "".join(i)
    lc_e=pd.read_csv(f'./output_japan/learning_curve/lc_nk_extra_{i}_0.csv')
    # idx = lc_i["test_loss"].idxmin()
    # lc_inter_extra_dict[i] = (idx,((1/5)*lc_i["test_loss"][idx]+(1/10)*lc_e["test_loss"][idx])/((1/5)+(1/10)))
    # lc_inter_extra_dict[i] = (lc_i["test_loss"][idx]+lc_e["test_loss"][idx])/2
    lc = ((1/5)*lc_i["test_loss"]+(1/10)*lc_e["test_loss"])/((1/5)+(1/10))
    lc_inter_extra_dict[i] = (lc.idxmin(),lc.min())
    # lc_inter_extra_dict[i] = (lc_i["test_loss"][10]+lc_e["test_loss"][10])/2
    # lc_inter_extra_dict[i] = (lc_i["test_loss"]+lc_e["test_loss"]).min()/2
    # lc_inter_extra_dict[i] = (idx,(lc_i["test_loss"].min()+lc_e["test_loss"].min())/2)
    # lc_inter_extra_dict[i] = (idx,(((1/5)*lc_i["test_loss"].min()+(1/10)*lc_e["test_loss"].min())/((1/5)+(1/10))))
lc_inter_extra_dict=sorted(lc_inter_extra_dict.items(),key= lambda x:x[1][1])[:5]
lc_inter_extra_dict
# %%
# hyper curie extra
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(25,40)
for method in ["dnn","nk"]:
    for i in ['basic_curie_onsen_depth800_grad','basic_curie_onsen_tishitsu_depth800_grad']:
        lc=pd.read_csv(f'./output_japan/learning_curve/lc_{method}_extra_{i}_0.csv')[:16]
        ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(['basic_curie_onsen_depth_grad','basic_curie_onsen_tishitsu_depth_grad'],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
lc_inter_dict
# %%
volcano=0
curie=0
onsen=0
depth=0
grad=0
tishitsu=0
for  a,b in lc_extra_dict:
    if "volcano" in a:
        volcano+=1
    if "curie" in a:
        curie+=1
    if "onsen" in a:
        onsen+=1
    if "depth" in a:
        depth+=1
    if "grad" in a:
        grad+=1
    if "tishitsu" in a:
        tishitsu+=1
volcano,curie,onsen,depth,grad,tishitsu
# %%
pd.read_csv("./input_japan/useful_data/database/GSJ_db_ja.csv").groupby(["x","y"],as_index=False).mean()
# %%
