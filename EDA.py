# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%
import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import pdist, squareform,cdist
import matplotlib.pyplot as plt
from tqdm import tqdm
import pykrige.variogram_models as vm
# %%
input_data = pd.read_csv('./input/useful_data/input_data.csv')
add_input_tishitsu=pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
input_data=input_data.merge(add_input_tishitsu,on=['x','y'],how='left')
input_data.isnull().sum()
# %%
sns.barplot(x='group_ja',y='t',data=input_data)
#%%
input_data
#%%
def grad_calc(df):
    xy_unique = df.groupby(['x', 'y'], as_index=False).mean()[
        ['x', 'y']].values
    for x, y in xy_unique:
        zt = df[(df['x'] == x) & (df['y'] == y)][['z', 't']]
        if zt.shape[0] == 1:
            zt = pd.DataFrame(np.array([[0, 0]]), columns=[
                              'z', 't']).append(zt)
        grad = np.polyfit(zt['z'], zt['t'], 1)[0]*1000
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad'] = grad
    return df
# %%


def grad_maxmin_calc(df):
    xy_unique = df.groupby(['x', 'y'], as_index=False).mean()[
        ['x', 'y']].values
    for x, y in xy_unique:
        zt = df[(df['x'] == x) & (
            df['y'] == y)][['z', 't']]
        if zt.shape[0] == 1:
            zt = pd.DataFrame(np.array([[0, 0]]), columns=[
                              'z', 't']).append(zt)
        zt=zt.sort_values('z')
        z_diff = np.diff(zt['z'])
        t_diff = np.diff(zt['t'])
        grad = (t_diff/z_diff)*1000
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_max'] = max(grad)
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_min'] = min(grad)
    return df


# %%
input_data = grad_calc(input_data)
input_data = grad_maxmin_calc(input_data)
input_data
# %%
sns.barplot(x='group_ja',y='t',data=input_data)
#%%
sns.pairplot(input_data)
#%%
input_data_g=input_data.groupby(['x','y'],as_index=False).mean()
# %%
xy_dis=squareform(pdist(input_data_g[['x','y']]))
#%%
t_vario=squareform(pdist(input_data_g['grad'].values.reshape(-1,1))**2)
# %%
sep=1100
max_dist=20001
sv_i=np.zeros(len(range(0,max_dist,sep)))
for i,value in enumerate(tqdm(range(0,max_dist,sep))):
    mask1=xy_dis>value
    mask2=xy_dis<value+sep
    mask=mask1*mask2
    res1=t_vario[mask]
    mask3=res1>0
    res2=(res1[mask3].mean())/2
    sv_i[i]=res2
sv_i

x=range(0,max_dist,sep)
plt.plot(x[:],sv_i[:],c='black',marker='o')
plt.plot(x[:],vm.spherical_variogram_model([0.01,200000,0.1],np.array(x))[:],c='red')


# %%
#%%
import pandas as pd
import numpy as np
# %%
profile_db=pd.read_csv('./input/useful_data/database/profile_db.csv')
geothermal_db=pd.read_csv('./input/useful_data/database/geothermal_db.csv')
GSJ_db=pd.read_csv('./input/useful_data/database/GSJ_db.csv')
profile_db['no']=0
geothermal_db['no']=1
GSJ_db['no']=2

# %%
input_data=pd.concat([profile_db,geothermal_db,GSJ_db]).reset_index(drop=True)
xy_input=input_data.groupby(['x','y'],as_index=False).mean()[['x','y','no']]

xy_input['x'],xy_input['y']=xy_input['x'].round(-1),xy_input['y'].round(-1)
xy_input[xy_input.duplicated(subset=['x','y'],keep=False)]['no'].value_counts()
#%%
profile_db['x'],profile_db['y']=profile_db['x'].round(-2),profile_db['y'].round(-2)
geothermal_db['x'],geothermal_db['y']=geothermal_db['x'].round(-2),geothermal_db['y'].round(-2)
GSJ_db['x'],GSJ_db['y']=GSJ_db['x'].round(-2),GSJ_db['y'].round(-2)
