#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
all_features_name_list=[]
features_name_list = ["volcano","curie","onsen","tishitsu_rank","tishitsu_ohe","depth","grad"]#,"tishitsu_ohe"
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
    if "tishitsu_rank" in all_f and "tishitsu_ohe" in all_f:
        continue
    for f in all_f:
        name+="_"+f
    all_name_list.append(name)
# all_name_list
#%%
for name in all_name_list:
    lc_list=[]
    for i in range(5):
        lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{name}_{i}.csv')['test_loss'].values
        lc_list.append(lc)
    lc=pd.DataFrame(np.vstack([np.arange(0,1001,100),np.vstack(lc_list).mean(axis=0)]).T,columns=['epoch','test_loss'])
    lc.to_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{name}.csv',index=False)
# %%

# inter
lc_inter_dict = {}
for i in all_name_list:
    if "tishitsu_rank" in i:
        continue
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    lc_inter_dict[i] = lc["test_loss"].min()#.min()#[10]#.min()#5]
    # lc_list=[]
    # hyper_list =[0.1915366765502127,
    # 0.19733788835890165,
    # 0.20986420867173736,
    # 0.19222422757939064,
    # 0.20903699883975763]
    # for j in range(5):
    # lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}_{j}.csv')['test_loss'].values
    #     lc_list.append(hyper_list[j]*lc)
    # lc_i = np.vstack(lc_list).sum(axis=0)
    # lc_inter_dict[i] = lc_i.min()
lc_inter_dict=sorted(lc_inter_dict.items(),key= lambda x:x[1])[:10]
lc_inter_dict

#%%
for key,value in lc_inter_dict:
    l=[]
    if "volcano" in key:
        l.append("活火山分布")
    if "curie" in key:
        l.append("キュリー点深度")
    if "tishitsu" in key:
        l.append("地質")
    if "onsen" in key:
        l.append("温泉")
    if "depth" in key:
        l.append("標高別地温")
    if "grad" in key:
        l.append("地温勾配")
    print(l)
#%%
# extra
lc_extra_dict = {}
for i in all_name_list:
    if "tishitsu_rank" in i:
        continue
    # if "depth" in i:
    #     i = i.split("depth")
    #     i.insert(1,"depth800")
    #     i = "".join(i)
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    lc_extra_dict[i] = lc["test_loss"].min()#.min()#[10]
lc_extra_dict=sorted(lc_extra_dict.items(),key= lambda x:x[1])[:10]
lc_extra_dict
# %%
# inter+extra
lc_inter_extra_dict = {}
for i in all_name_list:
    if "tishitsu_rank" in i:
        continue
    # lc_list=[]
    # hyper_list =[0.1915366765502127,
    # 0.19733788835890165,
    # 0.20986420867173736,
    # 0.19222422757939064,
    # 0.20903699883975763]
    # for j in range(5):
    #     lc=pd.read_csv(f'./output_japan/learning_curve/select_features/lc_dnn_inter_{i}_{j}.csv')['test_loss'].values
    #     lc_list.append(hyper_list[j]*lc)
    # lc_i = np.vstack(lc_list).sum(axis=0)
    # lc_i = pd.DataFrame(lc_i,columns=["test_loss"])
    lc_i=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    # if "depth" in i:
    #     i = i.split("depth")
    #     i.insert(1,"depth800")
    #     i = "".join(i)
    lc_e=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    # idx = lc_i["test_loss"].idxmin()
    # lc_inter_extra_dict[i] = (idx,((1/5)*lc_i["test_loss"][idx]+(1/10)*lc_e["test_loss"][idx])/((1/5)+(1/10)))
    # lc_inter_extra_dict[i] = (lc_i["test_loss"][idx]+lc_e["test_loss"][idx])/2
    # lc = ((1/5)*lc_i["test_loss"]+(1/10)*lc_e["test_loss"])/((1/5)+(1/10))
    lc = (lc_i["test_loss"]+lc_e["test_loss"])/2
    lc_inter_extra_dict[i] = (lc.idxmin(),lc.min())
    # lc_inter_extra_dict[i] = (lc.idxmin(),lc[10])
    # lc_inter_extra_dict[i] = (lc_i["test_loss"][10]+lc_e["test_loss"][10])/2
    # lc_inter_extra_dict[i] = (lc_i["test_loss"]+lc_e["test_loss"]).min()/2
    # lc_inter_extra_dict[i] = (idx,(lc_i["test_loss"].min()+lc_e["test_loss"].min())/2)
    # lc_inter_extra_dict[i] = (idx,(((1/5)*lc_i["test_loss"].min()+(1/10)*lc_e["test_loss"].min())/((1/5)+(1/10))))
lc_inter_extra_dict=sorted(lc_inter_extra_dict.items(),key= lambda x:x[1][1])[:10]
lc_inter_extra_dict
# %%
import math
from decimal import *
for key,value in lc_inter_extra_dict:
    l=[]
    if "volcano" in key:
        l.append("活火山分布")
    if "curie" in key:
        l.append("キュリー点深度")
    if "tishitsu" in key:
        l.append("地質")
    if "onsen" in key:
        l.append("温泉")
    if "depth" in key:
        l.append("標高別地温")
    if "grad" in key:
        l.append("地温勾配")
    for i,j in enumerate(l):
        if i==len(l)-1:
            print(j)
        else:
            print(j,end="，")
    # print("\n")
    # print(Decimal(str(value[1])).quantize(Decimal('.01'), rounding=ROUND_DOWN))
#%%
['basic_onsen_tishitsu_ohe_depth800_grad','basic_volcano_curie_onsen_depth800_grad','basic_volcano_curie_onsen_tishitsu_rank_depth800_grad']
#%%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(25,40)
lc_inter_extra_dict = {}
for i in ['basic_onsen_tishitsu_ohe_depth_grad','basic_volcano_curie_onsen_depth_grad','basic_volcano_curie_onsen_tishitsu_rank_depth_grad']:
    # if "volcano" in i:
    #     continue
    # lc_i=pd.read_csv(f'./output_japan/learning_curve/select_features/lc_dnn_inter_{i}.csv')[:11]
    lc_i=pd.read_csv(f'./output_japan/learning_curve/nk/lc_nk_inter_{i}.csv')
    # print(lc_i)
    if "depth" in i:
        i = i.split("depth")
        i.insert(1,"depth800")
        i = "".join(i)
    # lc_e=pd.read_csv(f'./output_japan/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    lc_e=pd.read_csv(f'./output_japan/learning_curve/lc_nk_extra_{i}_0.csv')#[::10].reset_index(drop=True)
    # print(lc_e)
    # idx = lc_i["test_loss"].idxmin()
    # lc_inter_extra_dict[i] = (idx,((1/5)*lc_i["test_loss"][idx]+(1/10)*lc_e["test_loss"][idx])/((1/5)+(1/10)))
    # lc_inter_extra_dict[i] = (lc_i["test_loss"][idx]+lc_e["test_loss"][idx])/2
    lc = ((1/5)*lc_i["test_loss"]+(1/10)*lc_e["test_loss"])/((1/5)+(1/10))
    print(lc_i["test_loss"],lc_e["test_loss"])
    ax.plot(lc_i['epoch'],lc)
    
    lc_inter_extra_dict[i] = (lc.idxmin(),lc.min())
    # lc_inter_extra_dict[i] = (lc_i["test_loss"][10]+lc_e["test_loss"][10])/2
    # lc_inter_extra_dict[i] = (lc_i["test_loss"]+lc_e["test_loss"]).min()/2
    # lc_inter_extra_dict[i] = (idx,(lc_i["test_loss"].min()+lc_e["test_loss"].min())/2)
    # lc_inter_extra_dict[i] = (idx,(((1/5)*lc_i["test_loss"].min()+(1/10)*lc_e["test_loss"].min())/((1/5)+(1/10))))
ax.legend(['basic_onsen_tishitsu_ohe_depth_grad','basic_volcano_curie_onsen_depth_grad','basic_volcano_curie_onsen_tishitsu_rank_depth_grad'],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
lc_inter_extra_dict=sorted(lc_inter_extra_dict.items(),key= lambda x:x[1][1])[:5]
lc_inter_extra_dict
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(20,35)
for i in ["basic_volcano_curie_onsen_tishitsu_rank_depth800_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
for i in ["basic_volcano_curie_onsen_tishitsu_rank_depth800_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan/learning_curve/lc_nk_extra_{i}_0.csv')[::10]
    ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["dnn","nk"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
import pandas as pd
pd.read_csv("./input_japan/tishitsu/database/add_grid_tishitsu_detail_idokeido_ja.csv")
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
#%%
fig, ax = plt.subplots()
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(25,35)
for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_inter_{i}.csv')[::2]
    ax.plot(lc['epoch'],lc['test_loss'],c="red")
#%%
fig, ax = plt.subplots()
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(15,40)
for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')#[::10]
    ax.plot(lc['epoch'],lc['test_loss'],c="red")
# %%
fig, ax = plt.subplots()
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(15,40)
for i in ["basic","basic_volcano","basic_curie","basic_tishitsu_ohe","basic_onsen","basic_depth","basic_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'],"red")
# for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
#     ax.plot(lc['epoch'],lc['test_loss'],c="red")
ax.legend(["BASEモデル","活火山モデル","キュリー点深度モデル","地質モデル","温泉モデル","標高別地温モデル","地温勾配モデル","ALLモデル"],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# for i in ["basic"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
#     ax.plot(lc['epoch'],lc['test_loss'],c="red")
# %%
for name in ["basic_grad_mm","basic_grad_only"]:
    lc_list=[]
    for i in range(5):
        lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{name}_{i}.csv')['test_loss'].values
        lc_list.append(lc)
    lc=pd.DataFrame(np.vstack([np.arange(0,1001,100),np.vstack(lc_list).mean(axis=0)]).T,columns=['epoch','test_loss'])
    lc.to_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{name}.csv',index=False)
#%%
fig, ax = plt.subplots()
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(15,40)
for i in ["basic","basic_grad","basic_grad_mm","basic_grad_only"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["basic","basic_grad","basic_grad_mm","basic_grad_only"])

# %%
fig, ax = plt.subplots()
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(15,40)
for i in ["basic","basic_grad","basic_grad_mm","basic_grad_only"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["basic","basic_grad","basic_grad_mm","basic_grad_only"])
# %%
