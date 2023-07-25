#%%
import pandas as pd
import numpy as np
#%%
lc_list=[]
name = "basic_curie_onsen_tishitsu_ohe_depth_grad"#"basic_volcano_curie_onsen_depth_grad"#"basic_onsen_tishitsu_ohe_depth_grad"
for i in range(5):
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_inter_{name}_{i}.csv')['test_loss'].values
    lc_list.append(lc)
lc=pd.DataFrame(np.vstack([np.arange(0,1001,10),np.vstack(lc_list).mean(axis=0)]).T,columns=['epoch','test_loss'])
# lc.to_csv(f'./output_japan_last/learning_curve/nk/lc_nk_inter_{name}.csv',index=False)
# %%
a=pd.read_csv('./output/voxler/est_nk_inter.csv')
a[a['h_z']==-5000]['t'].min()
# %%
name = "basic_volcano_curie_onsen_tishitsu_rank_depth_grad"#"basic_volcano_curie_onsen_depth_grad"#"basic_onsen_tishitsu_ohe_depth_grad"
t_list=[]
for i in range(5):
    df=pd.read_csv(f'./output_japan/voxler/nk/est_nk_output_{name}_{i}.csv')
    t=df['t'].values
    t_list.append(t)
df['t']=np.vstack(t_list).mean(axis=0)
# df.to_csv(f'./output_japan/voxler/nk/est_nk_output_{name}.csv',index=False)
# %%
a=pd.read_csv('./output/voxler/est_nk_inter_100.csv')
a[(a['h_z']>-3000) &(a['t']>374)]
# %%
a=pd.read_csv('./input/tishitsu/add_grid_tishitsu_pred.csv')
# a=a.groupby(['x','y'],as_index=False).min()
# a[a['h_z']<-5000].shape
a['lithology_ja'].unique().shape
# %%
