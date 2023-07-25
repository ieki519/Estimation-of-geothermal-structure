#%%
import pandas as pd
import numpy as np
#%%
lc_list=[]
for i in range(5):
    lc=pd.read_csv(f'./output/learning_curve/lc_nk_extra_addz_new_epoch2000{i}.csv')['test_loss'].values
    lc_list.append(lc)
lc=pd.DataFrame(np.vstack([np.arange(0,2001,100),np.vstack(lc_list).mean(axis=0)]).T,columns=['epoch','test_loss'])
# lc.to_csv('./output/learning_curve/lc_nk_extra_addz_new_epoch2000.csv',index=False)
# %%
a=pd.read_csv('./output/voxler/est_nk_inter.csv')
a[a['h_z']==-5000]['t'].min()
# %%
t_list=[]
for i in range(5):
    df=pd.read_csv(f'./output/voxler/est_nk_inter_addz_new_{i}_detail.csv')
    t=df['t'].values
    t_list.append(t)
df['t']=np.vstack(t_list).mean(axis=0)
# df.to_csv('./output/voxler/est_nk_inter_addz_new_detail.csv',index=False)
# %%
a=pd.read_csv('./output/voxler/est_nk_inter_100.csv')
a[(a['h_z']>-3000) &(a['t']>374)]
# %%
a=pd.read_csv('./input/tishitsu/add_grid_tishitsu_pred.csv')
# a=a.groupby(['x','y'],as_index=False).min()
# a[a['h_z']<-5000].shape
a['lithology_ja'].unique().shape
# %%
