#%%
from operator import index
import pandas as pd
import numpy as np
from tqdm import tqdm
# %%
def grid_albers(sep_xy,sep_z):
    #sep_xy(km),sep_z(m)#単位に注意
    
    #df1
    ido=range(950,1151,sep_xy)
    keido=range(2950,3051,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df1=pd.DataFrame(data,columns=['y','x','z'])
    #df2
    ido=range(800,1151,sep_xy)
    keido=range(3050,3251,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df2=pd.DataFrame(data,columns=['y','x','z'])
    #df3
    ido=range(1150,1451,sep_xy)
    keido=range(3100,3251,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df3=pd.DataFrame(data,columns=['y','x','z'])
    #df4
    ido=range(1050,1451,sep_xy)
    keido=range(3250,3351,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df4=pd.DataFrame(data,columns=['y','x','z'])
    #df5
    ido=range(1200,1751,sep_xy)
    keido=range(3350,3451,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df5=pd.DataFrame(data,columns=['y','x','z'])
    #df6
    ido=range(1200,2351,sep_xy)
    keido=range(3450,3551,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df6=pd.DataFrame(data,columns=['y','x','z'])
    #df7
    ido=range(1400,2201,sep_xy)
    keido=range(3550,3651,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df7=pd.DataFrame(data,columns=['y','x','z'])
    #df8
    ido=range(1400,2001,sep_xy)
    keido=range(3650,3751,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df8=pd.DataFrame(data,columns=['y','x','z'])
    #df9
    ido=range(2250,2751,sep_xy)
    keido=range(3300,3451,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df9=pd.DataFrame(data,columns=['y','x','z'])
    #df10
    ido=range(2450,2751,sep_xy)
    keido=range(3450,3651,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df10=pd.DataFrame(data,columns=['y','x','z'])
    #まとめ
    df=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10])
    df=df.loc[:,['x','y','z']]
    df['x']=df['x']-100000
    df['y']=df['y']-100000
    df=df.drop_duplicates(['x','y','z'])
    return df
# %%
xyz_grid=grid_albers(1,100)
#%%
# xyz_grid.to_csv('./grid_100.csv',index=False)
# xy=xyz_grid.groupby(['x','y'],as_index=False).min()
# xy.to_csv('../成形前データ/xy_grid_albers_100.csv',index=False)
#%%
xyz_grid[['x','y']].to_csv('./input/xy_grid_albers.csv',index=False)
# %%
xy_grid_wgs=pd.read_csv('./input/xy_grid_WGS.csv')
hyoukou=[]
#%%
# import re
# import requests
# # hyoukou=[]
# for ido, keido in tqdm(xy_grid_wgs.values):
#     url=f'https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php?lon={keido}&lat={ido}&outtype=JSON'
#     res = requests.get(url)
#     hyoukou.append(re.split('[:,]',res.text)[1])
# hyoukou
# %%
xy_grid_wgs['hyoukou']=[float(h) if not len(h)>6 else np.nan for h in hyoukou]
# xy_grid_wgs.to_csv('./input/xy_h_grid_wgs.csv',index=False)
# %%
xy_grid_albers=pd.read_csv('./input/xy_grid_albers.csv')
xy_grid_albers['hyoukou']=[float(h) if not len(h)>6 else np.nan for h in hyoukou]
# xy_grid_albers.to_csv('./input/xy_h_grid_albers.csv',index=False)
# %%
