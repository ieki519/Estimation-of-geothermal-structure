#%%
import pandas as pd
import numpy as np
from sqlalchemy import column
# %%
def grid_albers(sep_xy,sep_z):#sep_xy(km),sep_z(m)#単位に注意
    #sep_xy(km),sep_z(m)#単位に注意
    #df1
    ido=range(-200,300,sep_xy)
    keido=range(-2100,-1800,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df1=pd.DataFrame(data,columns=['y','x','z'])
    #df2
    ido=range(-100,300,sep_xy)
    keido=range(-1800,-1300,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df2=pd.DataFrame(data,columns=['y','x','z'])
    #df3
    ido=range(0,400,sep_xy)
    keido=range(-1300,-800,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df3=pd.DataFrame(data,columns=['y','x','z'])
    #df4
    ido=range(400,800,sep_xy)
    keido=range(-1000,-700,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df4=pd.DataFrame(data,columns=['y','x','z'])
    #df5
    ido=range(800,1000,sep_xy)
    keido=range(-900,-700,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df5=pd.DataFrame(data,columns=['y','x','z'])
    #df6
    ido=range(700,1200,sep_xy)
    keido=range(-700,-500,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df6=pd.DataFrame(data,columns=['y','x','z'])
    #df7
    ido=range(800,1000,sep_xy)
    keido=range(-500,-300,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df7=pd.DataFrame(data,columns=['y','x','z'])

    #まとめ
    df=pd.concat([df1,df2,df3,df4,df5,df6,df7])
    df=df.loc[:,['x','y','z']]
    df=df.drop_duplicates(['x','y','z'])
    
    return df
# %%
df=grid_albers(5,100)
df
# %%
# df.to_csv('./input_japan/grid_500_japan_detail.csv',index=False)
# %%
df.groupby(["x","y"],as_index=False).mean()[["x","y"]].to_csv('./input_japan/grid_500_japan_xy_detail.csv',index=False)
#%%
df=pd.read_cs("")
#%%
# albers=pd.read_csv('./input/xy_h_grid_albers.csv')
albers=pd.read_csv('./input/elevation/hyoukou_albers.csv')
albers
# %%
df=df.merge(albers,on=['x','y'],how='left').dropna()
df
#%%
df=df.rename(columns={'hyoukou':'h','z':'h_z'})
df
# %%
df['z']=df['h']-df['h_z']
df
# %%
df=df[['x','y','h','z','h_z']]
# %%
# df.to_csv('./input/useful_data/est_grid_100.csv',index=False)
# %%
df.dropna()
# %%
input_data = pd.read_csv("./input_japan/useful_data/input_data_ja.csv")
input_data_xy = input_data.groupby(["x","y"],as_index=False).mean()
input_data_xy
# %%
df_list= []
for x,y,h in input_data_xy[["x","y","h"]].values:
    df = pd.DataFrame(np.arange(1000,-5001,-10),columns=["h_z"])
    df["x"]=x
    df["y"]=y
    df["h"]=h
    df["z"]=df["h"]-df["h_z"]
    df=df[["x","y","h","z","h_z"]]
    df_list.append(df)
# %%
df=pd.concat(df_list).reset_index(drop=True)
# df.to_csv("./input_japan/useful_data/est_grid_input_xy_ja.csv",index=False)
# %%
#%%
import pandas as pd
import numpy as np
# %%
def grid_albers(sep_xy,sep_z):#sep_xy(km),sep_z(m)#単位に注意
    #sep_xy(km),sep_z(m)#単位に注意
    #df1
    ido=range(-200,300,sep_xy)
    keido=range(-2100,-1800,sep_xy)
    z=range(1500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df1=pd.DataFrame(data,columns=['y','x','z'])
    #df2
    ido=range(-100,300,sep_xy)
    keido=range(-1800,-1300,sep_xy)
    z=range(1500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df2=pd.DataFrame(data,columns=['y','x','z'])
    #df3
    ido=range(0,400,sep_xy)
    keido=range(-1300,-800,sep_xy)
    z=range(1500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df3=pd.DataFrame(data,columns=['y','x','z'])
    #df4
    ido=range(400,800,sep_xy)
    keido=range(-1000,-700,sep_xy)
    z=range(1500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df4=pd.DataFrame(data,columns=['y','x','z'])
    #df5
    ido=range(800,1000,sep_xy)
    keido=range(-900,-700,sep_xy)
    z=range(1500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df5=pd.DataFrame(data,columns=['y','x','z'])
    #df6
    ido=range(700,1200,sep_xy)
    keido=range(-700,-500,sep_xy)
    z=range(1500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df6=pd.DataFrame(data,columns=['y','x','z'])
    #df7
    ido=range(800,1000,sep_xy)
    keido=range(-500,-300,sep_xy)
    z=range(1500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df7=pd.DataFrame(data,columns=['y','x','z'])

    #まとめ
    df=pd.concat([df1,df2,df3,df4,df5,df6,df7])
    df=df.loc[:,['x','y','z']]
    df=df.drop_duplicates(['x','y','z'])
    
    return df
# %%
df=grid_albers(5,100)
df
# %%
df=df.rename(columns={"z":"h_z"})
df
#%%
xyh=pd.read_csv("./input_japan/useful_data/est_grid_detail_ja.csv").groupby(["x","y"],as_index=False).mean()[["x","y","h"]]
xyh
# %%
df=df.merge(xyh,on=["x","y"],how="left")
df
# %%
df=df.dropna().reset_index(drop=True)
df
# %%
df["z"]=df["h"]-df["h_z"]
df
# %%
df[["x","y","h","z","h_z"]].to_csv("./input_japan/useful_data/est_grid_detail_ja.csv",index=False)
#%%

#%%
import pandas as pd
import numpy as np
from sqlalchemy import column
# %%
def grid_albers(sep_xy,sep_z):#sep_xy(km),sep_z(m)#単位に注意
    #sep_xy(km),sep_z(m)#単位に注意
    #df1
    ido=range(-200,300,sep_xy)
    keido=range(-2100,-1800,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df1=pd.DataFrame(data,columns=['y','x','z'])
    #df2
    ido=range(-100,300,sep_xy)
    keido=range(-1800,-1300,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df2=pd.DataFrame(data,columns=['y','x','z'])
    #df3
    ido=range(0,400,sep_xy)
    keido=range(-1300,-800,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df3=pd.DataFrame(data,columns=['y','x','z'])
    #df4
    ido=range(400,800,sep_xy)
    keido=range(-1000,-700,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df4=pd.DataFrame(data,columns=['y','x','z'])
    #df5
    ido=range(800,1000,sep_xy)
    keido=range(-900,-700,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df5=pd.DataFrame(data,columns=['y','x','z'])
    #df6
    ido=range(700,1200,sep_xy)
    keido=range(-700,-500,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df6=pd.DataFrame(data,columns=['y','x','z'])
    #df7
    ido=range(800,1000,sep_xy)
    keido=range(-500,-300,sep_xy)
    z=range(500,-5001,-sep_z)
    data=[[i*1000,j*1000,k] for i in ido for j in keido for k in z]
    df7=pd.DataFrame(data,columns=['y','x','z'])

    #まとめ
    df=pd.concat([df1,df2,df3,df4,df5,df6,df7])
    df=df.loc[:,['x','y','z']]
    df=df.drop_duplicates(['x','y','z'])
    
    return df
# %%
df=grid_albers(10,250)
df
# %%
df=df.rename(columns={"z":"h_z"})
df
#%%
xyh=pd.read_csv("./input_japan/useful_data/est_grid_detail_ja.csv").groupby(["x","y"],as_index=False).mean()[["x","y","h"]]
xyh
# %%
df=df.merge(xyh,on=["x","y"],how="left")
df
# %%
df=df.dropna().reset_index(drop=True)
df
# %%
df["z"]=df["h"]-df["h_z"]
df
# %%
df=df[["x","y","h","z","h_z"]]
# %%
df.to_csv("./input_japan/useful_data/est_grid_sv250_ja.csv",index=False)
# %%
def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)

    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)

def torch_select_column_by_condition_2(data, index, condition):
    condition_data = data[:, index]
    mask1 = condition_data[:,index[0]].eq(condition[0])
    mask2 = condition_data[:,index[1]].eq(condition[1])
    mask=mask1*mask2
    if len(torch.nonzero(mask)) == 0:
        return torch.Tensor()
    indices = torch.squeeze(torch.nonzero(mask), 1)
    select = torch.index_select(data, 0, indices)
    return select

def create_d_z_list(ts_est,ts_est_xy):
    dist_z_list=[]
    for xy in ts_est_xy:
        pc=torch_select_column_by_condition_2(ts_est,[0,1],xy)
        h_z=pc[:,[2]]
        dist_z=torch.sqrt(pairwise_distances(h_z))
        dist_z_list.append(dist_z)
    return dist_z_list

#%%
# sv calc 
est_data_origin = df[['x', 'y', 'h_z']].values
est_data_origin_xy = df[['x', 'y', 'h_z']].groupby(["x","y"],as_index=False).mean()[["x","y"]].values
# to torch
est_data_origin = Variable(torch.from_numpy(
    est_data_origin).float()).to(device)
est_data_origin_xy = Variable(torch.from_numpy(
    est_data_origin_xy).double()).to(device)
#%%
dist_z_list = create_d_z_list(est_data_origin,est_data_origin_xy)

# %%
dist_z_list
# %%
len(dist_z_list)
# %%
dist_z_list[0].unique()
# %%
