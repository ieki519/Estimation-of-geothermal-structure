#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%

train_loss_list=[]
test_loss_list=[]
for i in range(3):
    lc=pd.read_csv(f'./output/learning_curve/lc_7_{i}.csv')
    epoch=lc['epoch']
    train_loss=lc['train_loss'].values
    test_loss=lc['test_loss'].values
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
train_loss_master=np.vstack(train_loss_list).mean(axis=0)
test_loss_master=np.vstack(test_loss_list).mean(axis=0)

train_loss_list=[]
test_loss_list=[]
for i in range(3):
    # if i==3 or 5:
    #     continue
    lc=pd.read_csv(f'./output/learning_curve/lc_10_{i}.csv')#f'C:/Users/Admin/Desktop/家木　卒論/年末ゼミ発表/アルベルスoutput/graph_642_{i}.csv'
    epoch=lc['epoch']
    train_loss=lc['train_loss'].values#'RMSE'
    test_loss=lc['test_loss'].values#'RMSE_VAL'
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
train_loss_master_c=np.vstack(train_loss_list).mean(axis=0)
test_loss_master_c=np.vstack(test_loss_list).mean(axis=0)

fig, ax = plt.subplots()
ax.grid()
# ax.plot(epoch,train_loss_master)
# epoch_=[0]+[e for e in epoch if e%100==0]
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
ax.plot(epoch,test_loss_master_c,color='black')
ax.plot(epoch,test_loss_master,color='red')
# ax.plot(2000,21.372042,'X',markersize=15,color='blue')
plt.rcParams["font.size"] = 20
ax.legend([u'Model_0',u'Model_1'],prop={'family':"MS Gothic"})
# %%
test_loss_master.shape,test_loss_master_c.shape
# %%

# %%
train_loss_list=[]
test_loss_list=[]
for i in range(3):
    lc=pd.read_csv(f'./output/learning_curve/lc_8_{i}.csv')
    epoch=lc['epoch']
    train_loss=lc['train_loss'].values
    test_loss=lc['test_loss'].values
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
train_loss_master=np.vstack(train_loss_list).mean(axis=0)
test_loss_master=np.vstack(test_loss_list).mean(axis=0)

train_loss_list=[]
test_loss_list=[]
for i in range(3):
    # if i==3 or 5:
    #     continue
    lc=pd.read_csv(f'./output/learning_curve/lc_11_{i}.csv')#f'C:/Users/Admin/Desktop/家木　卒論/年末ゼミ発表/アルベルスoutput/graph_641_{i}.csv'
    epoch=lc['epoch']
    train_loss=lc['train_loss'].values
    test_loss=lc['test_loss'].values
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
train_loss_master_c=np.vstack(train_loss_list).mean(axis=0)
test_loss_master_c=np.vstack(test_loss_list).mean(axis=0)

fig, ax = plt.subplots()
ax.grid()
# ax.plot(epoch,train_loss_master)
# epoch_=[0]+[e for e in epoch if e%100==0]
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
ax.plot(epoch,test_loss_master_c,color='black')
ax.plot(epoch,test_loss_master,color='red')
# ax.plot(2000,14.720051,'X',markersize=15,color='blue')
plt.rcParams["font.size"] = 20
ax.legend([u'Model_0',u'Model_1'],prop={'family':"MS Gothic"})

# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20

for i in ['_only_basic','_only_tishitsu','_only_onsen','_only_kmeans','']:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_inter.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([u'Basic',u'地質',u'温泉',u'k_means',u'All'],prop={'family':"MS Gothic"})
plt.show()
#%%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20

for i in ['_only_basic','_only_tishitsu','_only_onsen','_only_kmeans','']:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([u'Basic',u'地質',u'温泉',u'k_means',u'All'],prop={'family':"MS Gothic"})
plt.show()
#%%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20

lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_.csv')
ax.set_ylim(30,40)
ax.plot(lc['epoch'],lc['test_loss'],color='red')
lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra.csv')
ax.plot(lc['epoch'],lc['test_loss'],color='blue')
ax.legend([u'All',u'NK'],prop={'family':"MS Gothic"})
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20

lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter.csv')
ax.set_ylim(30,35)
ax.plot(lc['epoch'],lc['test_loss'],color='red')
lc=pd.read_csv(f'./output/learning_curve/lc_nk_inter.csv')
ax.plot(lc['epoch'],lc['test_loss'],color='blue')
ax.legend([u'All',u'NK'],prop={'family':"MS Gothic"})
plt.show()
# %%
# start
#%%
# only inter
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(32,40)
lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter_only_base.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="black")
for i in ["only_tishitsu","only_onsen","only_depth","only_grad"]:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([u'base',u'地質',u'温泉',u'標高別地温',u'地温勾配'],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
# ax.legend(['only_base','only_volcano','only_curie',"only_tishitsu","only_onsen","only_grad","only_depth"],prop={'family':"MS Gothic"})
plt.show()
# %%
# except inter 
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,40)
for i in ["all"]:#'only_base','except_volcano','except_curie',"except_tishitsu","except_onsen","except_grad","except_depth","all"]:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
lc=pd.read_csv(f'./output/learning_curve/lc_nk_inter100.csv')
ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(['only_base','except_volcano','except_curie',"except_tishitsu","except_onsen","except_grad","except_depth","all","nk"],prop={'family':"MS Gothic"})
plt.show()

#%%
# only extra 
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,60)
lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_only_base.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="black")
for i in ["only_tishitsu","only_onsen","only_depth","only_grad"]:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(['base','only_volcano','only_curie',"only_tishitsu","only_onsen","only_grad","only_depth"],prop={'family':"MS Gothic"})
plt.show()
# %%
# except extra 
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,50)
for i in ['only_base','except_volcano','except_curie',"except_tishitsu","except_onsen","except_grad","except_depth"]:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra100_0.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(['only_base','except_volcano','except_curie',"except_tishitsu","except_onsen","except_grad","except_depth"],prop={'family':"MS Gothic"})
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(25,40)
for i in ["all","all2"]:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra100_0.csv')
ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(['only_base','except_volcano','except_curie',"except_tishitsu","except_onsen","except_grad","except_depth","all","all2","nk"],prop={'family':"MS Gothic"})
plt.show()
# %%
# except inter 
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,40)
for i in ["all"]:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
lc=pd.read_csv(f'./output/learning_curve/lc_nk_inter100.csv')
ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["all","nk"],prop={'family':"MS Gothic"})
plt.show()

#%%
# except inter 
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(25,40)

# ax.set_xlim(1200,2000)
for i in ["all"]:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra100_0.csv')
ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["all","nk"],prop={'family':"MS Gothic"})
plt.show()
#%%
# 地質
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(25,40)
# ax.set_xlim(1200,2000)
color_list=["black","red"]
for i,value in enumerate(["only_base","only_tishitsu"]):
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_{value}.csv')
    ax.plot(lc['epoch'],lc['test_loss'],c=color_list[i])
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra100_0.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([u"base",u"地質"],prop={'family':"MS Gothic"})
plt.show()
#%%
# 温泉
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(25,40)
# ax.set_xlim(1200,2000)
for i,value in enumerate(["only_base","only_onsen"]):
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_{value}.csv')
    ax.plot(lc['epoch'],lc['test_loss'],c=color_list[i])
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra100_0.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([u"base",u"温泉"],prop={'family':"MS Gothic"})
plt.show()
#%%
# 地温勾配
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(25,40)
# ax.set_xlim(1200,2000)
color_list=["black","red","blue"]
for i,value in enumerate(["only_base","only_grad"]):
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_{value}.csv')
    ax.plot(lc['epoch'],lc['test_loss'],c=color_list[i])
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra100_0.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([u"base",u"地温勾配"],prop={'family':"MS Gothic"})
plt.show()
#%%
# depth
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(25,40)
# ax.set_xlim(1200,2000)
for i,value in enumerate(["only_base","only_depth"]):
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_{value}.csv')
    ax.plot(lc['epoch'],lc['test_loss'],c=color_list[i])
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra100_0.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([u"base",u"標高別地温"],prop={'family':"MS Gothic"})
plt.show()
#%%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,40)
lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter_only_base.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="black")
for i in ["except_tishitsu","except_onsen","except_depth","except_grad"]:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter_all.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="blue")
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
# ax.legend(['base',u"except_地質",u"except_温泉",u"except_標高別地温",u"except_地温勾配",u"all"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_xlim(1500,2000)
ax.set_ylim(30,60)

lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter_all.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="black")
lc=pd.read_csv(f'./output/learning_curve/lc_nk_inter100.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="red")
ax.legend([u"DNN",u"NK"],prop={'family':"MS Gothic"})
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_xlim(1500,2000)
# ax.set_ylim(27,60)

lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_all.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="black")
lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra100.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="red")
ax.legend([u"DNN",u"NK"],prop={'family':"MS Gothic"})

# ax.legend(['base',u"except_地質",u"except_温泉",u"except_標高別地温",u"except_地温勾配",u"all"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_xlim(1500,2000)
ax.set_ylim(30,60)

lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter_all.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="black")
lc=pd.read_csv(f'./output/learning_curve/lc_nk_inter100_1000.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="red")
ax.legend([u"DNN",u"NK"],prop={'family':"MS Gothic"})
plt.show()
#%%
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_xlim(1500,2000)
ax.set_ylim(30,60)

lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_all.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="black")
lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_stacking.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="red")
ax.legend([u"DNN",u"NK"],prop={'family':"MS Gothic"})
plt.show()
#%%
# except inter 
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,40)
for i in ['only_base','except_volcano','except_curie',"except_tishitsu","except_onsen","except_grad","except_depth","all","stacking"]:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(['only_base','except_volcano','except_curie',"except_tishitsu","except_onsen","except_grad","except_depth","all","stacking"],prop={'family':"MS Gothic"})
plt.show()

# %%
# except extra
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,40)
for i in ['only_base','except_volcano','except_curie',"except_tishitsu","except_onsen","except_grad","except_depth","all"]:
    lc=pd.read_csv(f'./output/learning_curve/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(['only_base','except_volcano','except_curie',"except_tishitsu","except_onsen","except_grad","except_depth","all"],prop={'family':"MS Gothic"})
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_xlim(1500,2000)
ax.set_ylim(25,40)

lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_all_0.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="black")
lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra100_0.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="blue")
lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra_addz_0.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="red")
ax.legend([u"DNN",u"NK",u"NK 改良"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_xlim(1500,2000)
ax.set_ylim(25,40)

lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_all.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="black")
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra_addz.csv')
# ax.plot(lc['epoch'],lc['test_loss'],c="blue")
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra_addz_new.csv')
# ax.plot(lc['epoch'],lc['test_loss'],c="red")
lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra_addz_new_epoch2000.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="red")
# ax.legend([u"DNN",u"NK",u"NK 改良"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_xlim(1500,2000)
# ax.set_ylim(25,40)
ax.set_ylim(30,40)

lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_all_cattishitsu.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="black")
lc=pd.read_csv(f'./output/learning_curve/lc_dnn_extra_except_tishitsu.csv')
ax.plot(lc['epoch'],lc['test_loss'],c="red")
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_inter_addz_new.csv')
# ax.plot(lc['epoch'],lc['test_loss'],c="red")
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra_addz_epoch2000.csv')
# ax.plot(lc['epoch'],lc['test_loss'],c="red")
# ax.legend([u"DNN",u"NK",u"NK 改良"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()

# %%
# except inter 
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
ax.set_ylim(30,40)
for i in ['except_volcano','except_curie',"except_tishitsu","except_onsen","except_grad","except_depth","all"]:
    lc=pd.read_csv(f'./output_new/learning_curve/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(['except_volcano','except_curie',"except_tishitsu","except_onsen","except_grad","except_depth","all"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
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
    lc=pd.read_csv(f'./output_new/learning_curve/lc_dnn_inter_{i}.csv')
    lc_inter_dict[i] = lc["test_loss"][20]
lc_inter_dict=sorted(lc_inter_dict.items(),key= lambda x:x[1])[:5]
lc_inter_dict
# %%
# extra
lc_extra_dict = {}
for i in all_name_list:
    # if "curie" in i:
    #     continue
    lc=pd.read_csv(f'./output_new/learning_curve/lc_dnn_extra80_{i}.csv')
    lc_extra_dict[i] = lc["test_loss"][20]
lc_extra_dict=sorted(lc_extra_dict.items(),key= lambda x:x[1])[:5]
lc_extra_dict
# %%
# inter+extra
lc_inter_extra_dict = {}
for i in all_name_list:
    # if "curie" in i:
    #     continue
    lc_i=pd.read_csv(f'./output_new/learning_curve/lc_dnn_inter_{i}.csv')
    lc_e=pd.read_csv(f'./output_new/learning_curve/lc_dnn_extra_{i}_0.csv')
    # lc_inter_extra_dict[i] = ((1/5)*lc_i["test_loss"][20]+(1/10)*lc_e["test_loss"][20])/((1/5)+(1/10))
    lc_inter_extra_dict[i] = (lc_i["test_loss"][20]+lc_e["test_loss"][20])/2
lc_inter_extra_dict=sorted(lc_inter_extra_dict.items(),key= lambda x:x[1])[:5]
lc_inter_extra_dict
# %%
len(all_name_list)
# %%
(64*25)/60
# %%
