#%% 
from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# %%
df=pd.read_csv('./output/voxler/est_nk_inter100.csv')
df_xy=df[df['h_z']==-5000]
x=df_xy['x']
y=df_xy['y']
t=df_xy['t']
# %%
sc = plt.scatter(x, y, vmin=0,vmax=500, c=t,marker='s',cmap=cm.jet,s=5)
plt.colorbar(sc)
plt.show()
#%%