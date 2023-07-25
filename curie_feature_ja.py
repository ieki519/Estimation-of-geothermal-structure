#%%
import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import pdist, squareform,cdist
import matplotlib.pyplot as plt
from tqdm import tqdm
import pykrige.variogram_models as vm
#%%
##feature
curie=pd.read_csv('./input_japan/curie_point/database/curie_albers_ja.csv')
curie
#%%
xy_dis=squareform(pdist(curie[['x','y']]))
#%%
t_vario=squareform(pdist(curie['curie'].values.reshape(-1,1))**2)
# %%
sep=1100
max_dist=200001
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
# %%
x=range(0,max_dist,sep)
plt.plot(x[:],sv_i[:],c='black',marker='o')
plt.plot(x[:],vm.spherical_variogram_model([10,200000,0.5],np.array(x))[:],c='red')

#%%
curie=curie.values
okd = OrdinaryKriging(curie[:, 0], curie[:, 1], curie[:, 2],
                                                 variogram_model='spherical',variogram_parameters={'sill':10,'range':200000,'nugget':0.5})  
curie
# %%
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data=input_data.groupby(['x','y'],as_index=False).mean()[['x','y']]
input_xy=input_data.values
input_predict=okd.execute('points',input_xy[:, 0], input_xy[:, 1])[0].data
input_data['curie']=input_predict
# input_data.to_csv('./input_japan/curie_point/add_input_curie_ja.csv',index=False)
#%%
est_data = pd.read_csv('./input_japan/useful_data/est_grid_500_ja.csv')
est_data=est_data.groupby(['x','y'],as_index=False).mean()[['x','y']]
est_xy=est_data.values.astype('float')
input_predict=okd.execute('points',est_xy[:, 0], est_xy[:, 1])[0].data
est_data['curie']=input_predict
# est_data.to_csv('./input_japan/curie_point/add_grid_curie_ja.csv',index=False)
# %%
onsen = pd.read_csv('./input_japan/onsen/database/onsen_xyh_ja.csv')
onsen=onsen.groupby(['x','y'],as_index=False).mean()[['x','y']]
onsen_xy=onsen.values.astype('float')
onsen_predict=okd.execute('points',onsen_xy[:, 0], onsen_xy[:, 1])[0].data
onsen['curie']=onsen_predict
# onsen.to_csv('./input_japan/onsen/add_onsen_curie_ja.csv',index=False)
#%%
## grid_curie

# curie_xyh_zt=pd.read_csv('./input/curie_point/curie_albers.csv')
# curie_xyh_zt=curie_xyh_zt.rename(columns={'curie':'h_z'})
# curie_xyh_zt['h_z']=-curie_xyh_zt['h_z']*1000
# curie_xyh_zt['t']=580
# curie_xyh_zt
# %%
grid_curie=pd.read_csv('./input_japan/curie_point/add_grid_curie_ja.csv')
grid_curie=grid_curie.rename(columns={'curie':'z'})
grid_curie['z']=grid_curie['z']*1000
grid_curie['t']=580
grid_curie
# %%
est_data = pd.read_csv('./input_japan/useful_data/est_grid_500_ja.csv')
est_data_xyh=est_data.groupby(['x','y','h'],as_index=False).mean()[['x','y','h']]
est_data_xyh
# %%
grid_curie_master=est_data_xyh.merge(grid_curie,how='left',on=['x','y'])
grid_curie_master['h_z']=grid_curie_master['h']-grid_curie_master['z']
grid_curie_master=grid_curie_master[['x','y','h','z','h_z','t']]
grid_curie_master
# %%
# grid_curie_master.to_csv('./input_japan/curie_point/grid_curie_ja.csv',index=False)
#%%
import pandas as pd
import numpy as np
# %%
def grid_albers(sep_xy,sep_z):#sep_xy(km),sep_z(m)#単位に注意
    #sep_xy(km),sep_z(m)#単位に注意
    #df1
    ido=range(-200,300,sep_xy)
    keido=range(-2100,-1800,sep_xy)
    z=range(500,-20001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df1=pd.DataFrame(data,columns=['y','x','z'])
    #df2
    ido=range(-100,300,sep_xy)
    keido=range(-1800,-1300,sep_xy)
    z=range(500,-20001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df2=pd.DataFrame(data,columns=['y','x','z'])
    #df3
    ido=range(0,400,sep_xy)
    keido=range(-1300,-800,sep_xy)
    z=range(500,-20001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df3=pd.DataFrame(data,columns=['y','x','z'])
    #df4
    ido=range(400,800,sep_xy)
    keido=range(-1000,-700,sep_xy)
    z=range(500,-20001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df4=pd.DataFrame(data,columns=['y','x','z'])
    #df5
    ido=range(800,1000,sep_xy)
    keido=range(-900,-700,sep_xy)
    z=range(500,-20001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df5=pd.DataFrame(data,columns=['y','x','z'])
    #df6
    ido=range(700,1200,sep_xy)
    keido=range(-700,-500,sep_xy)
    z=range(500,-20001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df6=pd.DataFrame(data,columns=['y','x','z'])
    #df7
    ido=range(800,1000,sep_xy)
    keido=range(-500,-300,sep_xy)
    z=range(500,-20001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df7=pd.DataFrame(data,columns=['y','x','z'])

    #まとめ
    df=pd.concat([df1,df2,df3,df4,df5,df6,df7])
    df=df.loc[:,['x','y','z']]
    df=df.drop_duplicates(['x','y','z'])
    df=df.rename(columns={'z':'h_z'})
    return df
#%%
df=grid_albers(10,500)
df
#%%
grid_curie=pd.read_csv('./input_japan/curie_point/grid_curie_ja.csv')
grid_curie=grid_curie.rename(columns={'h_z':'curie_point'})
del grid_curie['z'],grid_curie['t']
grid_curie
# %%
df=df.merge(grid_curie,on=['x','y'],how='left').dropna()
df['z']=df['h']-df['h_z']
df=df.reset_index(drop=True)
df
# %%
df.groupby(['x','y'],as_index=False).mean()
# %%
curie_580izyou=df[df['curie_point']-df['h_z']>0].reset_index(drop=True)
curie_580ika=df[df['curie_point']-df['h_z']<0].reset_index(drop=True)
curie_580izyou=curie_580izyou[['x','y','h','z','h_z']]
curie_580ika=curie_580ika[['x','y','h','z','h_z']]
curie_580izyou.to_csv('./input_japan/curie_point/grid_curie_580izyou_ja.csv',index=False)
curie_580ika.to_csv('./input_japan/curie_point/grid_curie_580ika_ja.csv',index=False)
# %%
curie_580ika
# %%
