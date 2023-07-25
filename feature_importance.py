#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
for i in ['grad','grad_mm','grad_mmhz']:
    lc=pd.read_csv(f'./output_japan/learning_curve/features/lc_dnn_inter_feature_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(['grad','grad_mm','grad_mmhz'],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
# perm extra grad
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
plt.rcParams["font.size"] = 20
# ax.set_ylim(30,40)
# ax.set_ylim(40,60)
# lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_basic_grad.csv')
# ax.plot(lc['epoch'],lc['test_loss'],"red")
for i in ['grad','grad_mm','grad_mmhz']:
    lc=pd.read_csv(f'./output_japan/learning_curve/features/lc_dnn_extra_feature_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/unit_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(['grad','grad_mm','grad_mmhz'],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
# %%
