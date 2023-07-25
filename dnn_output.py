#%%
import pandas as pd
import numpy as np
#%%
all_features_name_list=[]
features_name_list = ["volcano","curie","onsen","tishitsu","depth","grad"]
N=len(features_name_list)
for i in range(2**N):
    A=[]
    for j in range(N):
        if ((i>>j)&1):
            A.append(features_name_list[j])
    all_features_name_list.append(A)
features_dict = {
                "volcano":['volcano'],
                "curie":['curie'],
                "onsen":['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
                "tishitsu":['age_a', 'age_b','age'],
                "depth":[],#extraは500
                "grad":[],
                }
#%%
for all_f in all_features_name_list:
    name = "basic"
    for f in all_f:
        name += "_"+f 
    
    lc_list=[]
    for i in range(5):
        lc=pd.read_csv(f'./output_japan/learning_curve/lc_dnn_inter_{name}_{i}.csv')['test_loss'].values
        lc_list.append(lc)
    lc=pd.DataFrame(np.vstack([np.arange(0,2001,100),np.vstack(lc_list).mean(axis=0)]).T,columns=['epoch','test_loss'])
    lc.to_csv(f'./output_japan/learning_curve/lc_dnn_inter_{name}.csv',index=False)
    
    t_list=[]
    for i in range(5):
        df=pd.read_csv(f'./output_japan/voxler/est_dnn_inter_{name}_{i}.csv')
        t=df['t'].values
        t_list.append(t)
    df['t']=np.vstack(t_list).mean(axis=0)
    df.to_csv(f'./output_japan/voxler/est_dnn_inter_{name}.csv',index=False)
# %%
