#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
all_features_name_list = ["volcano","curie","onsen","tishitsu","tishitsu_rank","depth","grad"]
features_dict = {
                "volcano":['volcano'],
                "curie":['curie'],
                "onsen":['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
                "tishitsu":['age','group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩'],
                "tishitsu_rank":["nan",'age',"group_rank"],
                "depth":["depth0","depth500","depth1000"],#extraは800
                "grad":["nan",'grad','grad_max','grad_min','grad_max_h_z','grad_min_h_z'],
                }
#%%
for all_f in all_features_name_list:
    for except_f in features_dict[all_f]:
        name = "perm_"+all_f
        lc_list=[]
        for i in range(5):
            lc=pd.read_csv(f'./output_japan/learning_curve/perm/lc_dnn_inter_{name}_except_{except_f}_{i}.csv')['test_loss'].values
            lc_list.append(lc)
        lc=pd.DataFrame(np.vstack([np.arange(0,2001,100),np.vstack(lc_list).mean(axis=0)]).T,columns=['epoch','test_loss'])
        lc.to_csv(f'./output_japan/learning_curve/perm/lc_dnn_inter_{name}_except_{except_f}.csv',index=False)

#%%
# perm inter onsen 
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,40)
# ax.set_ylim(40,60)
lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_basic_onsen.csv')
ax.plot(lc['epoch'],lc['test_loss'],"red")
for i in ['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion']:
    lc=pd.read_csv(f'./output_japan/learning_curve/perm/lc_dnn_inter_perm_onsen_except_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["basic",'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(25,40)
# ax.set_ylim(40,60)
lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_extra_basic_onsen.csv')
ax.plot(lc['epoch'],lc['test_loss'],"red")
for i in ['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion']:
    lc=pd.read_csv(f'./output_japan/learning_curve/perm/lc_dnn_extra_perm_onsen_except_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["basic",'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
# --------------------------------------------------

# perm inter tishitsu
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(34,40)
# ax.set_ylim(40,60)
lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_basic_tishitsu.csv')
ax.plot(lc['epoch'],lc['test_loss'],"red")
for i in ['age','group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩']:
    lc=pd.read_csv(f'./output_japan/learning_curve/perm/lc_dnn_inter_perm_tishitsu_except_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["basic",'age','group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩'],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,50)
# ax.set_ylim(40,60)
lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_extra_basic_tishitsu.csv')
ax.plot(lc['epoch'],lc['test_loss'],"red")
for i in ['age','group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩']:
    lc=pd.read_csv(f'./output_japan/learning_curve/perm/lc_dnn_extra_perm_tishitsu_except_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["basic",'age','group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩'],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%

# --------------------------------------------------
# perm inter tishitsu_rank
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(34,40)
# ax.set_ylim(40,60)
lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_basic_tishitsu.csv')
ax.plot(lc['epoch'],lc['test_loss'],"red")

for i in ["nan",'age','group_rank']:
    lc=pd.read_csv(f'./output_japan/learning_curve/perm/lc_dnn_inter_perm_tishitsu_rank_except_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["nan",'age','group_rank'],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,50)
# ax.set_ylim(40,60)
# lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_extra_basic_tishitsu.csv')
# ax.plot(lc['epoch'],lc['test_loss'],"red")

for i in ["nan",'age','group_rank']:
    lc=pd.read_csv(f'./output_japan/learning_curve/perm/lc_dnn_extra_perm_tishitsu_rank_except_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["nan",'age','group_rank'],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
#%%
# --------------------------------------------------
# perm inter depth
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(35,40)
# ax.set_ylim(40,60)
lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_basic_depth.csv')
ax.plot(lc['epoch'],lc['test_loss'],"red")
for i in ["depth0","depth500","depth1000"]:
    lc=pd.read_csv(f'./output_japan/learning_curve/perm/lc_dnn_inter_perm_depth_except_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["basic","depth0","depth500","depth1000"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(30,60)
# ax.set_ylim(40,60)
lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_extra_basic_depth.csv')
ax.plot(lc['epoch'],lc['test_loss'],"red")
for i in ["depth0","depth500","depth800"]:
    lc=pd.read_csv(f'./output_japan/learning_curve/perm/lc_dnn_extra_perm_depth_except_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["basic","depth0","depth500","depth800"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%

# --------------------------------------------------
# perm inter grad
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,40)
# ax.set_ylim(40,60)
# lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_basic_grad.csv')
# ax.plot(lc['epoch'],lc['test_loss'],"red")
for i in ["nan",'grad','grad_max','grad_min',"grad_max_h_z","grad_min_h_z"]:
    lc=pd.read_csv(f'./output_japan/learning_curve/perm/lc_dnn_inter_perm_grad_except_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["basic",'grad','grad_max','grad_min',"grad_max_h_z","grad_min_h_z"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(30,60)
# ax.set_ylim(40,60)
# lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_extra_basic_grad.csv')
# ax.plot(lc['epoch'],lc['test_loss'],"red")
for i in ["nan",'grad','grad_max','grad_min',"grad_max_h_z","grad_min_h_z"]:
    lc=pd.read_csv(f'./output_japan/learning_curve/perm/lc_dnn_extra_perm_grad_except_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["nan",'grad','grad_max','grad_min',"grad_max_h_z","grad_min_h_z"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
