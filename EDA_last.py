#%%
from turtle import Turtle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import pykrige.variogram_models as vm
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# %%
def extra_split(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.1, shuffle=False)
    return train_data, test_data

def preprocess_input(df):
    df['t'] = np.where(df['t'].values <= 0, 0.1, df['t'].values)
    add_input_volcano = pd.read_csv('./input_japan/volcano/add_input_volcano_ja.csv')
    add_input_curie = pd.read_csv('./input_japan/curie_point/add_input_curie_ja.csv')
    add_input_tishitsu = pd.read_csv('./input_japan/tishitsu/add_input_tishitsu_pred_ja.csv')
    add_input_onsen = pd.read_csv('./input_japan/onsen/add_input_onsen_ja.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])

    # df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df=pd.concat([df,tmp],axis=1)
    
    df=df.drop(['symbol','symbol_freq','formationAge_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)#, 'group_ja'
    df['age']=(df['age_a']+df['age_b'])/2

    return df
#%%
input_data=pd.read_csv("./input_japan/useful_data/input_data_ja.csv")
input_data.describe()
#%%
input_data=preprocess_input(input_data)
train_data_d=input_data.groupby(["x","y"],as_index=False).min()
train_data_d[train_data_d.h_z<=-3000]
#%%
767,546,246

xy = input_data.groupby(["x","y"],as_index=False).mean()
xy[['group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩']].sum()
#%%
train,test = extra_split(input_data)
train.describe()
#%%
xy = input_data.groupby(["x","y"],as_index=False).mean()[["x","y"]]
xy.shape
#%%
sns.set(font='Yu Gothic')
sns.set_context("talk")
fig,ax = plt.subplots()
ax.scatter(x="t",y="h_z",data=input_data,s=5)
ax.set_xlabel("地温（℃）")
ax.set_ylabel("標高（m）")
ax.set_title("地温と標高の散布図")
fig.savefig(f"./修論用画像/地温と標高の散布図.png",dpi=300,bbox_inches = 'tight')

#%%
sns.set(font='Yu Gothic')
sns.set_context("talk")

fig,ax = plt.subplots()
ax.hist(input_data.t,bins=100)
print(input_data.t.skew())
ax.set_xlabel("地温（℃）")
ax.set_ylabel("Frequency")
ax.set_title("地温のヒストグラム")
fig.savefig(f"./修論用画像/地温のヒストグラム.png",dpi=300,bbox_inches = 'tight')

#%%
sns.set(font='Yu Gothic')
sns.set_context("talk")
input_data.t=input_data.t+1
# input_data = input_data[input_data.t>=1]
fig,ax = plt.subplots()
ax.hist(np.log10(input_data.t),bins=100)
print(np.log10(input_data.t).skew())
ax.set_xlabel("地温（℃）")
ax.set_ylabel("Frequency")
ax.set_title("地温のヒストグラム(対数変換後)")
fig.savefig(f"./修論用画像/地温のヒストグラム対数.png",dpi=300,bbox_inches = 'tight')
#%%

#%%
sns.set(font='Yu Gothic')
sns.set_context("talk")
# input_data = input_data[input_data.t>10]
fig,ax = plt.subplots()
ax.hist(input_data.z,bins=100)
ax.set_xlabel("深度(m)")
ax.set_ylabel("Frequency")
ax.set_title("深度のヒストグラム")
# fig.savefig(f"./修論用画像/深度のヒストグラム.png",dpi=300,bbox_inches = 'tight')

#%%
sns.set(font='Yu Gothic')
a=sns.jointplot(x="t",y="h_z",data=input_data, edgecolor="blue",facecolor='None')
a.set_axis_labels("地温（℃）","標高（m）")

# plt.savefig(f"./修論用画像/地温標高.png",dpi=300,bbox_inches='tight')
#%%
train,test = extra_split(input_data)
sns.set(font='Yu Gothic')
fig,ax = plt.subplots()
ax.plot(train.t,train.h_z,".",c="green")
ax.plot(test.t,test.h_z,".",c="red")
# ax.plot([0,330],[-895,-895],c="red",lw=4)
ax.set_xlabel("地温（℃）")
ax.set_ylabel("標高（m）")
ax.legend(["学習データ","検証データ"],fontsize=14)
plt.savefig(f"./修論用画像/extra_split.png",dpi=300,bbox_inches='tight')
#%%
for i in [27,52,66,63]:#range(xy.shape[0]):
    print(i)
    number = i
    x=xy.x[number]
    y=xy.y[number]
    train=input_data[(input_data.x==x)&(input_data.y==y)]
    
    # df = est_data[(est_data.x==x)&(est_data.y==y)]
    # abcd = np.polyfit(df.h_z.values,df.t.values,8)
    # est_data.loc[(est_data.x==x)&(est_data.y==y),"t"]=np.poly1d(abcd)(df.h_z.values)
    
    # test = est_data[(est_data.x==x)&(est_data.y==y)]#[::30].reset_index(drop=True)
    # min=train.h_z.min()
    # test = test[test.h_z<min]
    fig, ax = plt.subplots()
    # ax.set_xlim(0,500)
    # ax.set_ylim(-2000,0)
    ax.set_xlabel('Temperature(℃)') 
    ax.set_ylabel('Elevation(m)')
    # ax.grid()
    plt.rcParams["font.size"] = 12
    ax.plot(train["t"],train["h_z"],c="black")
    if i==27:
        ax.text(-0.2, 1.05, "①", ha='center',fontsize=25, transform=ax.transAxes)
    elif i==52:
        ax.text(-0.2, 1.05, "②", ha='center',fontsize=25, transform=ax.transAxes)
    elif i==66:
        ax.text(-0.2, 1.05, "③", ha='center',fontsize=25, transform=ax.transAxes)
    elif i==63:
        ax.text(-0.2, 1.05, "④", ha='center',fontsize=25, transform=ax.transAxes)
    # ax.plot(test["t"],test["h_z"],c="red")
    plt.show()
    fig.savefig(f"./修論用画像/温度検層_{i}.png",bbox_inches = 'tight')
    if i==100:
        break
    
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

def grad_maxmin_calc(df):
    xy_unique = df.groupby(['x', 'y'], as_index=False).mean()[
        ['x', 'y']].values
    for x, y in xy_unique:
        zt = df[(df['x'] == x) & (
            df['y'] == y)][['z', 't']]
        h = df[(df['x'] == x) & (
            df['y'] == y)][['h']].values.max()
        zt.z = zt.z.round(-2)
        zt = zt.groupby("z",as_index=False).mean()
        if zt.shape[0] == 1:
            zt = pd.DataFrame(np.array([[0, 0]]), columns=[
                              'z', 't']).append(zt)
        zt = zt.sort_values('z')
        z_diff = np.diff(zt['z'])
        t_diff = np.diff(zt['t'])
        hz = h-zt[:-1].z.values-z_diff/2
        
        grad = (t_diff/z_diff)*1000
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_max'] = max(grad)
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_min'] = min(grad)
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_max_h_z'] = hz[np.argmax(grad)]
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_min_h_z'] = hz[np.argmin(grad)]
    return df
#%%
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)
input_data = grad_calc(input_data)
input_data = grad_maxmin_calc(input_data)
input_data
#%%
#%%
volcano = input_data.copy()
volcano=volcano.groupby(["x","y"],as_index=False).mean()
volcano.volcano = volcano.volcano/1000
sns.set(font='Yu Gothic')
sns.set_context("talk")

# volcano = volcano.rename(columns={"volcano":"火山からの距離（km）","t":"地温（℃）"})
a=sns.jointplot(x="t",y="volcano",data=volcano)#,kind="reg")
a.set_axis_labels("地温（℃）","活火山からの距離（km）")
plt.savefig(f"./修論用画像/地温火山からの距離.png",dpi=300,bbox_inches='tight')
#%%
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
ax.hist(volcano.curie,bins=26)
#%%
# learning inter
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 50
ax.set_ylim(30,60)
ax.set_title("内挿評価")

diff_list=[]
for i in ["","_volcano"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_basic{i}.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
print(diff_list[0]-diff_list[1])
ax.legend(["BASEモデル","活火山モデル"],prop={'family':"MS Gothic"})#,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
fig.savefig(f"./修論用画像/火山_inter.png",dpi=300,bbox_inches = 'tight')
plt.show()
#%%
# learning extra
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 50
ax.set_ylim(30,80)
ax.set_title("外挿評価")
diff_list=[]
for i in ["","_volcano"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_basic{i}_0.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
    
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
print(diff_list[0]-diff_list[1])

ax.legend(["BASEモデル","活火山モデル"],prop={'family':"MS Gothic"})#,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
fig.savefig(f"./修論用画像/火山_extra.png",dpi=300,bbox_inches = 'tight')

plt.show()
# %%
curie = input_data.copy()
curie=curie.groupby(["x","y"],as_index=False).max()
curie.curie = curie.curie
sns.set(font='Yu Gothic')
sns.set_context("talk")
# curie = curie.rename(columns={"curie":"火山からの距離（km）","t":"地温（℃）"})
a=sns.jointplot(x="t",y="curie",data=curie)#,kind="reg")#,kind="reg")
a.set_axis_labels("地温（℃）","キュリー点深度（km）")
plt.savefig(f"./修論用画像/地温キュリー点深度.png",dpi=300,bbox_inches='tight')

# %%
#%%
# learning inter
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 50
# ax.set_ylim(25,40)
ax.set_ylim(30,60)

diff_list=[]
ax.set_title("内挿評価")
for i in ["","_curie"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_basic{i}.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[1])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","キュリー点深度モデル"],prop={'family':"MS Gothic"})#,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
fig.savefig(f"./修論用画像/キュリー点深度_inter.png",dpi=300,bbox_inches = 'tight')
plt.show()
print(6.410861082843724-5.462925019305892)
#%%
# learning extra
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 50
ax.set_ylim(30,80)
ax.set_title("外挿評価")
diff_list=[]
for i in ["","_curie"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_basic{i}_0.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[1])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","キュリー点深度モデル"],prop={'family':"MS Gothic"})#,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
fig.savefig(f"./修論用画像/キュリー点深度_extra.png",dpi=300,bbox_inches = 'tight')

plt.show()
#%%
onsen = input_data.copy()
onsen = onsen.rename(columns={"t":"地温","Temp":"泉温","anion":"AI",'SO4':"SO$_{4}$",'HCO3':"HCO$_{3}$"})
onsen=onsen.groupby(["x","y"],as_index=False).max()
onsen=onsen[["地温",'泉温', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', "SO$_{4}$", "HCO$_{3}$", 'AI']]
cmap = sns.color_palette("seismic", 200)
sns.heatmap(onsen.corr(), vmax=1, vmin=-1,square=True, cmap=cmap)
plt.savefig(f"./修論用画像/温泉相関係数.png",dpi=300,bbox_inches='tight')
# %%
#%%
# learning inter
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 50
ax.set_ylim(30,60)
diff_list=[]
ax.set_title("内挿評価")

for i in ["","_onsen"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_basic{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[1])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","温泉モデル"],prop={'family':"MS Gothic"})#,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
fig.savefig(f"./修論用画像/温泉_inter.png",dpi=300,bbox_inches = 'tight')
plt.show()
#%%

# result_df.to_csv('./input_japan/onsen/database/onsen_pred_result.csv')
# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
result_df=pd.read_csv('./input_japan/onsen/database/onsen_pred_result.csv').rename(columns={'Unnamed: 0':'Model'})

for key in ['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3','anion']:
    fig, ax = plt.subplots()
    # 
    sns.set()
    sns.set(font='Yu Gothic')
    sns.set_context("talk")
    # plt.rcParams["font.size"] = 18
    if key=="Temp":
        ax.set_title("泉温")
    elif key =="SO4":
        ax.set_title("SO$_{4}$")
    elif key =="HCO3":
        ax.set_title("HCO$_{3}$")
    elif key =="anion":
        ax.set_title("アニオンインデックス（AI）")
    else:
        ax.set_title(key)
            
    ax.set_xlabel('Algorithm')
    if key=='Temp':
        ax.set_ylabel('RMSE(℃)')  
    elif key=='pH' or key=='anion':
        ax.set_ylabel('RMSE')
    else:
        ax.set_ylabel('RMSE(mg/kg)')
    bar=ax.bar(['XGB','LGBM','RF','OK',"ST①","ST②"],result_df[key])
    bar[result_df[key].idxmin()].set_color("red")
    # ax.grid()
    fig.savefig(f"./修論用画像/{key}_est.png",dpi=300,bbox_inches='tight')
    plt.show()

#%%
onsen = pd.read_csv('./input_japan/onsen/database/onsen_xyh_ja.csv')
onsen = onsen[~(onsen['h'] == '-----')]
Temp = onsen.loc[:, ['x', 'y', 'Temp']].dropna().astype(float)
pH = onsen.loc[:, ['x', 'y', 'pH']].dropna().astype(float)
Na = onsen.loc[:, ['x', 'y', 'Na']].dropna().astype(float)
K = onsen.loc[:, ['x', 'y', 'K']].dropna().replace(
    ['<0.1', '<1.0'], 0).astype(float)
Ca = onsen.loc[:, ['x', 'y', 'Ca']].dropna().replace(['<1.0'], 0).astype(float)
Mg = onsen.loc[:, ['x', 'y', 'Mg']].dropna().replace(
    ['<0.1', '<1.0', '<0.01', '<0.02', 'TR'], 0).astype(float)
Cl = onsen.loc[:, ['x', 'y', 'Cl']].dropna().replace(['TR'], 0).astype(float)
SO4 = onsen.loc[:, ['x', 'y', 'SO4']].dropna().replace(
    ['<5', '<0.05', 'N.D.', '<0.1', '<1', '<1.0', 'TR'], 0).astype(float)
HCO3 = onsen.loc[:, ['x', 'y', 'HCO3']].dropna().replace(
    ['TR', 'N.D.', '<0.6', '<10', '<0.01', '<0.1'], 0).astype(float)
anion = onsen.loc[:, ['x', 'y', 'SO4', 'Cl', 'HCO3']].dropna().replace(
    ['<5', '<0.05', 'N.D.', '<0.1', '<1', '<1.0', 'TR', '<0.6', '<10', '<0.01'], 0).astype(float)
anion_SO4 = anion['SO4'].values
anion_Cl = anion['Cl'].values
anion_HCO3 = anion['HCO3'].values
anion_index = 0.5*(anion_SO4/(anion_Cl+anion_SO4) +
                   (anion_Cl+anion_SO4)/(anion_Cl+anion_SO4+anion_HCO3))
anion['anion'] = anion_index
anion = anion.loc[:, ['x', 'y', 'anion']]
def variogram(df, sep=11000, max_dist=500001, parameters={'sill': 9, 'range': 200000, 'nugget': 0.5}, how='gaussian'):
    fig, ax = plt.subplots()
    # 
    sns.set()
    sns.set(font='Yu Gothic')
    sns.set_context("talk")
    
    
    xy_dis = squareform(pdist(df.iloc[:, [0, 1]]))
    t_vario = squareform(pdist(df.iloc[:, 2].values.reshape(-1, 1))**2)
    sv_i = np.zeros(len(range(0, max_dist, sep)))
    sill_, range_, nugget_ = parameters['sill'], parameters['range'], parameters['nugget']
    for i, value in enumerate(tqdm(range(0, max_dist, sep))):
        mask1 = xy_dis > value
        mask2 = xy_dis < value+sep
        mask = mask1*mask2
        res1 = t_vario[mask]
        mask3 = res1 > 0
        res2 = (res1[mask3].mean())/2
        sv_i[i] = res2
    x = np.arange(0, max_dist, sep)[::10]/1000
    ax.plot(x, sv_i[::10], c='black', marker='o')
    if how == 'gaussian':
        ax.plot(x, vm.gaussian_variogram_model(
            [sill_, range_/1000, nugget_], x), c='red')
    elif how == 'spherical':
        ax.plot(x, vm.spherical_variogram_model(
            [sill_, range_/1000, nugget_], x), c='red')
    print(df.columns[2])
    
    ax.set_title("泉温")
    ax.set_xlabel('Distance between data(km)') 
    ax.set_ylabel('Semivariogram')
    # ax.set_title("泉温(Temp)[::10]","pH[:16]","Na[::2]","K[:13]","Ca[:15]","Mg[:]","Cl[:]","SO$_{4}$[:]","HCO$_{3}$[:]","アニオンインデックス（AI）")
    fig.savefig(f"./修論用画像/泉温_sv.png",dpi=300,bbox_inches='tight')
    plt.show()
if 1:
    variogram(Temp, sep=1000, max_dist=200001, parameters={
        'sill': 300, 'range': 75000, 'nugget': 175}, how='spherical')
    # variogram(pH, sep=1000, max_dist=20001, parameters={
    #     'sill': 1.75-0.5, 'range': 2500, 'nugget': 0.5})
    # variogram(Na, sep=1000, max_dist=40001, parameters={
    #     'sill': 2250000-800000, 'range': 7500, 'nugget': 800000}, how='spherical')
    # variogram(K, sep=5000, max_dist=100001, parameters={
    #     'sill': 60000, 'range': 25000, 'nugget': 20000})
    # variogram(Ca, sep=50, max_dist=1001, parameters={
    #     'sill': 150000-50000, 'range': 150, 'nugget': 50000}, how='spherical')
    # variogram(Mg, sep=11000, max_dist=400001, parameters={
    #     'sill': 16000, 'range': 180000, 'nugget': 12000})
    # variogram(Cl, sep=1000, max_dist=30001, parameters={
    #     'sill': 9000000-3500000, 'range': 8000, 'nugget': 3500000}, how='spherical')
    # variogram(SO4, sep=1000, max_dist=25001, parameters={
    #     'sill': 220000, 'range': 2000, 'nugget': 80000})
    # variogram(HCO3, sep=11000, max_dist=400001, parameters={
    #     'sill': 500000, 'range': 50000, 'nugget': 200000})
    # variogram(anion, sep=1000, max_dist=20001, parameters={
    #     'sill': 0.03-0.007, 'range': 5000, 'nugget': 0.007})
#%%

#%%
#%%
# learning extra
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 50
ax.set_ylim(25,80)

ax.set_title("外挿評価")

diff_list=[]
for i in ["","_onsen"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_basic{i}_0.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[1])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","温泉モデル"],prop={'family':"MS Gothic"})#,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
fig.savefig(f"./修論用画像/温泉_extra.png",dpi=300,bbox_inches = 'tight')
plt.show()
# %%
PI_onsen=pd.read_csv("./output_japan_last/PI/basic_onsen_PI_inter.csv")
fig, ax = plt.subplots(figsize=(9,4))
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
print(PI_onsen["name"].values)
ax.set_title("内挿評価")

bar=ax.bar(["泉温","Cl","Na","Anion\nIndex","pH","Ca","SO$_{4}$","HCO$_{3}$","Mg","K"],PI_onsen["PI"].values)
ax.set_ylabel("Permutation Importance")
plt.show()
fig.savefig(f"./修論用画像/PI_onsen_inter.png",dpi=300,bbox_inches='tight')

#%%
PI_onsen=pd.read_csv("./output_japan_last/PI/basic_onsen_PI_extra.csv")
fig, ax = plt.subplots(figsize=(9,4))
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
print(PI_onsen["name"].values)
ax.set_title("外挿評価")

bar=ax.bar(["泉温","Na","Cl","Anion\nIndex","pH","Mg","HCO$_{3}$","Ca","SO$_{4}$","K"],PI_onsen["PI"].values)
ax.set_ylabel("Permutation Importance")
plt.show()
fig.savefig(f"./修論用画像/PI_onsen_extra.png",dpi=300,bbox_inches='tight')
#%%
tishitsu = input_data.copy().groupby(["x","y"],as_index=False).mean()[["x","y","h","z","h_z","t"]]
tishitsu =preprocess_input(tishitsu)
sns.set(font='Yu Gothic')
sns.set_context("talk")
# sns.set_context("paper")
# volcano = volcano.rename(columns={"volcano":"火山からの距離（km）","t":"地温（℃）"})
a=sns.jointplot(x="t",y="age",data=tishitsu)#,kind="reg")
a.set_axis_labels("地温（℃）","形成年代（万年）")
# plt.yscale('log')
plt.savefig(f"./修論用画像/地温形成年代.png",dpi=300,bbox_inches='tight')
# %%
sns.set_context("talk")
b=sns.catplot(x="group_ja", y="t", kind="box",data=tishitsu,order=["火成岩","堆積岩","その他","付加体","変成岩"])
b.set_axis_labels("岩石区分","地温（℃）")
plt.savefig(f"./修論用画像/岩種box.png",dpi=300,bbox_inches='tight')

#%%
# learning inter
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 50
ax.set_ylim(30,60)
ax.set_title("内挿評価")
diff_list=[]
for i in ["","_tishitsu_ohe"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_basic{i}.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[1])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","地質モデル"],prop={'family':"MS Gothic"})
fig.savefig(f"./修論用画像/地質ohe_inter.png",dpi=300,bbox_inches='tight')
plt.show()
#%%
# learning extra
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 50
# ax.set_ylim(25,40)
ax.set_ylim(30,80)
ax.set_title("外挿評価")

diff_list=[]
for i in ["","_tishitsu_ohe"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_basic{i}_0.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[1])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","地質モデル"],prop={'family':"MS Gothic"})
fig.savefig(f"./修論用画像/地質ohe_extra.png",dpi=300,bbox_inches='tight')
plt.show()
#%%
# %%
PI_tishitsu=pd.read_csv("./output_japan_last/PI/basic_tishitsu_ohe_PI_inter.csv")
fig, ax = plt.subplots(figsize=(7,5))
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
print(PI_tishitsu["name"].values)
ax.set_title("内挿評価")
bar=ax.bar(["堆積岩","火成岩","形成年代","付加体","その他","変成岩"],PI_tishitsu["PI"].values)
ax.set_ylabel("Permutation Importance")
plt.show()
fig.savefig(f"./修論用画像/PI_tishitsu_ohe_inter.png",dpi=300,bbox_inches='tight')
#%%
PI_tishitsu=pd.read_csv("./output_japan_last/PI/basic_tishitsu_ohe_PI_extra.csv")
fig, ax = plt.subplots(figsize=(7,5))
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
print(PI_tishitsu["name"].values)
ax.set_title("外挿評価")

bar=ax.bar(["火成岩","堆積岩","付加体","形成年代","その他","変成岩"],PI_tishitsu["PI"].values)
ax.set_ylabel("Permutation Importance")
plt.show()
fig.savefig(f"./修論用画像/PI_tishitsu_ohe_extra.png",dpi=300,bbox_inches='tight')
#%%
depth = input_data.copy()
depth = depth.rename(columns={"t":"地温"})
depth=depth.groupby(["x","y"],as_index=False).mean()
depth=depth[["地温",'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion']]
cmap = sns.color_palette("coolwarm", 200)
sns.heatmap(depth.corr(), vmax=1, vmin=-1,square=True, cmap=cmap)
# %%
#%%
# learning inter
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
ax.set_title("内挿評価")

# plt.rcParams["font.size"] = 50
# ax.set_ylim(25,40)
diff_list=[]
for i in ["","_depth"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_basic{i}.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[1])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","標高別地温モデル"],prop={'family':"MS Gothic"})
fig.savefig(f"./修論用画像/標高別地温_inter.png",dpi=300,bbox_inches='tight')
plt.show()
#%%
# learning extra
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
ax.set_title("外挿評価")

# plt.rcParams["font.size"] = 50
ax.set_ylim(25,80)
diff_list=[]
for i in ["","_depth"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_basic{i}_0.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[1])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","標高別地温モデル"],prop={'family':"MS Gothic"})
fig.savefig(f"./修論用画像/標高別地温_extra.png",dpi=300,bbox_inches='tight')
plt.show()
# %%
PI_depth=pd.read_csv("./output_japan_last/PI/basic_depth_PI_inter.csv")
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
ax.set_title("内挿評価")


print(PI_depth["name"].values)
bar=ax.bar(["標高-500mの\n地温分布","標高0mの\n地温分布","標高-1000mの\n地温分布"],PI_depth["PI"].values)
ax.set_ylabel("Permutation Importance")
plt.show()
fig.savefig(f"./修論用画像/PI_depth_inter.png",dpi=300,bbox_inches='tight')
#%%
PI_depth=pd.read_csv("./output_japan_last/PI/basic_depth_PI_extra.csv")
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
ax.set_title("外挿評価")


print(PI_depth["name"].values)
bar=ax.bar(["標高-500mの\n地温分布","標高0mの\n地温分布"],PI_depth["PI"].values)
ax.set_ylabel("Permutation Importance")
plt.show()
fig.savefig(f"./修論用画像/PI_depth_extra.png",dpi=300,bbox_inches='tight')
#%%
def variogram(df, sep=11000, max_dist=500001, parameters={'sill': 9, 'range': 200000, 'nugget': 0.5}, how='gaussian'):
    fig, ax = plt.subplots()
    # 
    sns.set()
    sns.set(font='Yu Gothic')
    sns.set_context("talk")
    xy_dis = squareform(pdist(df.iloc[:, [0, 1]]))
    t_vario = squareform(pdist(df.iloc[:, 2].values.reshape(-1, 1))**2)
    sv_i = np.zeros(len(range(0, max_dist, sep)))
    sill_, range_, nugget_ = parameters['sill'], parameters['range'], parameters['nugget']
    for i, value in enumerate(tqdm(range(0, max_dist, sep))):
        mask1 = xy_dis > value
        mask2 = xy_dis < value+sep
        mask = mask1*mask2
        res1 = t_vario[mask]
        mask3 = res1 > 0
        res2 = (res1[mask3].mean())/2
        sv_i[i] = res2
    x = np.arange(0, max_dist, sep)[:]/1000
    ax.plot(x, sv_i[:], c='black', marker='o')
    if how == 'gaussian':
        ax.plot(x, vm.gaussian_variogram_model(
            [sill_, range_/1000, nugget_], x), c='red')
    elif how == 'spherical':
        ax.plot(x, vm.spherical_variogram_model(
            [sill_, range_/1000, nugget_], x), c='red')
    print(df.columns[2])
    
    ax.set_title("地温（標高-800m）")
    ax.set_xlabel('Distance between data(km)') 
    ax.set_ylabel('Semivariogram')
    # ax.set_title("泉温(Temp)[::10]","pH[:16]","Na[::2]","K[:13]","Ca[:15]","Mg[:]","Cl[:]","SO$_{4}$[:]","HCO$_{3}$[:]","アニオンインデックス（AI）")
    fig.savefig(f"./修論用画像/depth800_sv.png",dpi=300,bbox_inches='tight')
    plt.show()
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data.h_z=input_data.h_z.round(-1)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()
input_data_0=input_data[input_data.h_z==0]
input_data_500=input_data[input_data.h_z==-500]
input_data_800=input_data[input_data.h_z==-800]
input_data_1000=input_data[input_data.h_z==-1000]
input_data_1500=input_data[input_data.h_z==-1500] 

# variogram(input_data_0[["x","y","t"]], sep=1100, max_dist=20001, parameters={
#             'sill': 2500-500, 'range': 6000, 'nugget': 500}, how='spherical')
# variogram(input_data_500[["x","y","t"]], sep=1100, max_dist=20001, parameters={
#         'sill': 3000-500, 'range': 5000, 'nugget': 500}, how='spherical')
variogram(input_data_800[["x","y","t"]], sep=1100, max_dist=20001, parameters={
        'sill': 2000-250, 'range': 3000, 'nugget': 250}, how='spherical')
# variogram(input_data_1000[["x","y","t"]], sep=1100, max_dist=20001, parameters={
#         'sill': 3000-1250, 'range': 20000, 'nugget': 1250}, how='spherical')
# variogram(input_data_1000[["x","y","t"]], sep=5000, max_dist=100001, parameters={
#         'sill': 3000-1250, 'range': 20000, 'nugget': 1250}, how='gaussian')
#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
result_df=pd.read_csv('./input_japan/depth/database/depth_pred_result.csv').rename(columns={'Unnamed: 0':'Model'})
result_df
#%%
for key in ["0","500","1000"]:
    fig, ax = plt.subplots()
    # 
    sns.set()
    sns.set(font='Yu Gothic')
    sns.set_context("talk")
    # plt.rcParams["font.size"] = 18
    if key=="0":
        ax.set_title("地温（標高0m）")
    elif key =="500":
        ax.set_title("地温（標高-500m）")
    elif key =="1000":
        ax.set_title("地温（標高-1000m）")
            
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('RMSE(℃)')  

    bar=ax.bar(['XGB','LGBM','RF','OK',"ST①","ST②"],result_df[key])
    bar[result_df[key].idxmin()].set_color("red")
    # ax.grid()
    fig.savefig(f"./修論用画像/標高-{key}_est.png",dpi=300,bbox_inches='tight')
    plt.show()


#%%
grad = input_data.copy()
grad = grad.rename(columns={"t":"地温"})
grad=grad.groupby(["x","y"],as_index=False).mean()
grad=grad[["地温",'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion']]
cmap = sns.color_palette("coolwarm", 200)
sns.heatmap(grad.corr(), vmax=1, vmin=-1,square=True, cmap=cmap)
# %%
result_df=pd.read_csv('./input_japan/grad/database/grad_pred_result.csv').rename(columns={'Unnamed: 0':'Model'})
#%%
for key in ['grad','grad_max','grad_min','grad_max_h_z','grad_min_h_z']:
    fig, ax = plt.subplots()
    # 
    sns.set()
    sns.set(font='Yu Gothic')
    sns.set_context("talk")
    # plt.rcParams["font.size"] = 18
    if key=="grad":
        ax.set_title("地温勾配")
        ax.set_ylabel('RMSE(℃/km)')  
    elif key =="grad_max":
        ax.set_title("地温変化の最大値(標高100mごと)")
        ax.set_ylabel('RMSE(℃/km)')  
    elif key =="grad_min":
        ax.set_title("地温変化の最小値(標高100mごと)")
        ax.set_ylabel('RMSE(℃/km)')  
    elif key =="grad_max_h_z":
        ax.set_title("最大地温変化における標高")
        ax.set_ylabel('RMSE(m)')  
    elif key =="grad_min_h_z":
        ax.set_title("最小地温変化における標高")
        ax.set_ylabel('RMSE(m)')  
            
    ax.set_xlabel('Algorithm')
    

    bar=ax.bar(['XGB','LGBM','RF',"ST①"],result_df[key])
    bar[result_df[key].idxmin()].set_color("red")
    # ax.grid()
    fig.savefig(f"./修論用画像/地温勾配_{key}_est.png",dpi=300,bbox_inches='tight')
    plt.show()
#%%
#%%
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)
input_data = grad_calc(input_data)
input_data = grad_maxmin_calc(input_data)
input_data
#%%
xy = input_data.groupby(["x","y"],as_index=False).mean()
xy.describe()
#%%
grad = xy[["t","h",'grad','grad_max','grad_min','grad_max_h_z','grad_min_h_z']]
cmap = sns.color_palette("seismic", 200)
sns.heatmap(grad.corr(), vmax=1, vmin=-1,square=True, cmap=cmap)
# plt.savefig(f"./修論用画像/温泉相関係数.png",dpi=300,bbox_inches='tight')
#%%
# learning inter
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 50
# ax.set_ylim(25,40)
ax.set_title("内挿評価")
diff_list=[]
for i in ["","_grad_only","_grad"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_basic{i}.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[1])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","地温勾配モデル①","地温勾配モデル②"],prop={'family':"MS Gothic"})
fig.savefig(f"./修論用画像/地温勾配_inter.png",dpi=300,bbox_inches='tight')
plt.show()
#%%
# learning extra
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
ax.set_title("外挿評価")

# plt.rcParams["font.size"] = 50
ax.set_ylim(25,80)
diff_list=[]
for i in ["","_grad_only","_grad"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_basic{i}_0.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[1])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","地温勾配モデル①","地温勾配モデル②"],prop={'family':"MS Gothic"})
fig.savefig(f"./修論用画像/地温勾配_extra.png",dpi=300,bbox_inches='tight')
plt.show()
# %%
# %%
PI_grad=pd.read_csv("./output_japan_last/PI/basic_grad_PI_inter.csv")
fig, ax = plt.subplots(figsize=(6.5,4))#figsize=(9,5)
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
ax.set_title("内挿評価")
ax.set_ylabel("Permutation Importance")
print(PI_grad["name"].values)
bar=ax.bar(["地温勾配","地温勾配\n(max)","標高\n(max)","標高\n(min)","地温勾配\n(min)"],PI_grad["PI"].values)
plt.show()
fig.savefig(f"./修論用画像/PI_grad_inter.png",dpi=300,bbox_inches='tight')
#%%
PI_grad=pd.read_csv("./output_japan_last/PI/basic_grad_PI_extra.csv")
fig, ax = plt.subplots(figsize=(6.5,4))
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
ax.set_title("外挿評価")
ax.set_ylabel("Permutation Importance")
print(PI_grad["name"].values)
bar=ax.bar(["地温勾配","地温勾配\n(max)","標高\n(max)","地温勾配\n(min)","標高\n(min)"],PI_grad["PI"].values)
plt.show()
fig.savefig(f"./修論用画像/PI_grad_extra.png",dpi=300,bbox_inches='tight')
#%%
#%%
# learning inter
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 50
# ax.set_ylim(25,40)
for i in ["","_volcano","_curie","_onsen","_depth","_grad"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_basic{i}.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","地温勾配モデル"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
#%%
# learning extra
fig, ax = plt.subplots()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
# ax.grid()
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 50
# ax.set_ylim(25,40)
for i in ["","_depth800"]:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_basic{i}_0.csv')[:11]
    ax.plot(lc['epoch'],lc['test_loss'])
# lc=pd.read_csv(f'./output/learning_curve/curie_hyper/lc_nk_inter100.csv')
# ax.plot(lc['epoch'],lc['test_loss'])
ax.legend(["BASEモデル","地温勾配モデル"],prop={'family':"MS Gothic"},bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()
#%%
onsen = pd.read_csv('./input_japan/onsen/database/onsen_xyh_ja.csv')
onsen = onsen[~(onsen['h'] == '-----')]
onsen
# %%
onsen[["x","y","h","Ca"]].dropna().replace(['<1.0'], 0).astype(float).to_csv("a.csv",index=False)
# %%
# %%
fig, ax = plt.subplots()
# ax.grid()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")

ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
ax.set_title("内挿評価")
# plt.rcParams["font.size"] = 20
# ax.set_ylim(15,40)
diff_list=[]
for i in ["basic","basic_volcano","basic_curie","basic_tishitsu_ohe","basic_onsen","basic_depth","basic_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])

for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'],"red")
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[-1])
# for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
#     ax.plot(lc['epoch'],lc['test_loss'],c="red")
# ax.legend(["BASEモデル","活火山モデル","キュリー点深度モデル","地質モデル","温泉モデル","標高別地温モデル","地温勾配モデル","ALLモデル"],ncol = 4,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# for i in ["basic"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
#     ax.plot(lc['epoch'],lc['test_loss'],c="red")
# fig.savefig(f"./修論用画像/ALL_inter.png",dpi=300,bbox_inches='tight')

# %%

# %%
fig, ax = plt.subplots()
# ax.grid()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")

ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 20
# ax.set_ylim(15,40)
ax.set_title("外挿評価")
diff_list=[]
for i in ["basic","basic_volcano","basic_curie","basic_tishitsu_ohe","basic_onsen","basic_depth","basic_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'],"red")
    diff_list.append(lc["test_loss"][10])
print(diff_list[0]-diff_list[-1])
# for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
#     ax.plot(lc['epoch'],lc['test_loss'],c="red")
ax.legend(["BASEモデル","活火山モデル","キュリー点深度モデル","地質モデル","温泉モデル","標高別地温モデル","地温勾配モデル","ALLモデル"],ncol = 4,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# for i in ["basic"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
#     ax.plot(lc['epoch'],lc['test_loss'],c="red")
# fig.savefig(f"./修論用画像/ALL_extra.png",dpi=300,bbox_inches='tight')

# %%
PI_all=pd.read_csv("./output_japan_last/PI/basic_volcano_curie_onsen_tishitsu_ohe_depth_grad_PI_inter.csv")
fig, ax = plt.subplots(figsize=(8,5))#figsize=(9,5)
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
ax.set_title("内挿評価")
ax.set_ylabel("Permutation Importance")

print(PI_all["name"].values)
bar=ax.bar(["標高別\n地温","地温勾配","温泉","地質","キュリー点\n深度","活火山"],PI_all["PI"].values)
plt.show()
fig.savefig(f"./修論用画像/PI_all_inter.png",dpi=300,bbox_inches='tight')
#%%
PI_all=pd.read_csv("./output_japan_last/PI/basic_volcano_curie_onsen_tishitsu_ohe_depth_grad_PI_extra.csv")
fig, ax = plt.subplots(figsize=(8,5))
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")
ax.set_title("外挿評価")
ax.set_ylabel("Permutation Importance")

print(PI_all["name"].values)
bar=ax.bar(["標高別\n地温","地温勾配","地質","温泉","キュリー点\n深度","活火山"],PI_all["PI"].values)
plt.show()
fig.savefig(f"./修論用画像/PI_all_extra.png",dpi=300,bbox_inches='tight')
# %%

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
result_df=pd.read_csv('./input_japan/sv/pred_sv_result.csv').rename(columns={'Unnamed: 0':'Model'})
result_df
#%%
for key in [250,500,750,1000]:
    fig, ax = plt.subplots()
    # 
    sns.set()
    sns.set(font='Yu Gothic')
    sns.set_context("talk")
    # plt.rcParams["font.size"] = 18
    if key==250:
        ax.set_title("$\gamma(250\mathrm{m})$")
    elif key ==500:
        ax.set_title("$\gamma(500\mathrm{m})$")
    elif key == 750:
        ax.set_title("$\gamma(750\mathrm{m})$")
    elif key ==1000:
        ax.set_title("$\gamma(1000\mathrm{m})$")
        
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('RMSE(℃$^{2}$)')  

    bar=ax.bar(['XGB','LGBM','RF',"ST①"],result_df[f"sv_{key}"])
    bar[result_df[f"sv_{key}"].idxmin()].set_color("red")
    # ax.grid()
    fig.savefig(f"./修論用画像/sv_{key}_est.png",dpi=300,bbox_inches='tight')
    plt.show()
# %%
fig, ax = plt.subplots()
# ax.grid()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")

ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
ax.set_title("内挿評価")
# plt.rcParams["font.size"] = 20
ax.set_ylim(20,40)
diff_list=[]
# for i in ["basic"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
#     ax.plot(lc['epoch'],lc['test_loss'])
#     diff_list.append(lc["test_loss"][10])

for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'],c="cyan")
    diff_list.append(lc["test_loss"][10])

for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_inter_{i}.csv')[::10]
    ax.plot(lc['epoch'],lc['test_loss'],c="blue")
    diff_list.append(lc["test_loss"][100])
    
for i in ["basic_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'],c="magenta")
    diff_list.append(lc["test_loss"][10])

for i in ["basic_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_inter_{i}.csv')[::10]
    ax.plot(lc['epoch'],lc['test_loss'],c="red")
    diff_list.append(lc["test_loss"][100])
    
ax.legend(["ALLモデル(DNN)","ALLモデル(NK)","SELECTモデル(DNN)","SELECTモデル(NK)"])

print(diff_list[0]-diff_list[-1])
# ax.legend(["BASEモデル","活火山モデル","キュリー点深度モデル","地質モデル","温泉モデル","標高別地温モデル","地温勾配モデル","ALLモデル"],ncol = 4,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# for i in ["basic"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
#     ax.plot(lc['epoch'],lc['test_loss'],c="red")
# fig.savefig(f"./修論用画像/NK_inter.png",dpi=300,bbox_inches='tight')

# %%
lc["test_loss"]
# %%
fig, ax = plt.subplots()
# ax.grid()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")

ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 20
ax.set_ylim(15,40)
# ax.set_ylim(15,60)
ax.set_title("外挿評価")
diff_list=[]
# for i in ["basic"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
#     ax.plot(lc['epoch'],lc['test_loss'])
#     diff_list.append(lc["test_loss"][10])

for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'],c="cyan")
    diff_list.append(lc["test_loss"][10])

for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
    ax.plot(lc['epoch'],lc['test_loss'],c="blue")
    diff_list.append(lc["test_loss"][100])
    
for i in ["basic_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'],"magenta")
    diff_list.append(lc["test_loss"][10])

for i in ["basic_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
    ax.plot(lc['epoch'],lc['test_loss'],c="red")
    diff_list.append(lc["test_loss"][100])
    
print(diff_list[0]-diff_list[-1])
ax.legend(["ALLモデル(DNN)","ALLモデル(NK)","SELECTモデル(DNN)","SELECTモデル(NK)"],ncol = 2,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# ax.legend(["BASEモデル","活火山モデル","キュリー点深度モデル","地質モデル","温泉モデル","標高別地温モデル","地温勾配モデル","ALLモデル"],ncol = 4,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# for i in ["basic"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
#     ax.plot(lc['epoch'],lc['test_loss'],c="red")
# fig.savefig(f"./修論用画像/NK_extra.png",dpi=300,bbox_inches='tight')

# %%
fig, ax = plt.subplots()
# ax.grid()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")

ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 20
ax.set_ylim(15,40)
# ax.set_ylim(15,60)
ax.set_title("外挿評価")
diff_list=[]
# for i in ["basic"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
#     ax.plot(lc['epoch'],lc['test_loss'])
#     diff_list.append(lc["test_loss"][10])

for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'],c="cyan")
    diff_list.append(lc["test_loss"][10])

for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
    ax.plot(lc['epoch'],lc['test_loss'],c="blue")
    diff_list.append(lc["test_loss"][100])
    
for i in ["basic_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'],"magenta")
    diff_list.append(lc["test_loss"][10])

for i in ["basic_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
    ax.plot(lc['epoch'],lc['test_loss'],c="red")
    diff_list.append(lc["test_loss"][100])
    
print(diff_list[0]-diff_list[-1])
ax.legend(["ALLモデル(DNN)","ALLモデル(NK)","SELECTモデル(DNN)","SELECTモデル(NK)"])
# %%
a=pd.read_csv("./output_japan_last/voxler/nk/vox_est_nk_output_basic_volcano_curie_onsen_tishitsu_ohe_depth_grad_detail.csv")
a
# %%
a[a.t==a.t.max()].to_csv("./a.csv",index=False)
a[a.t==a.t.max()]

# %%
a=pd.read_csv("./input_japan/useful_data/input_data_ja.csv")
a=a.groupby(["x","y"],as_index=False).min()
a[a.h_z<=-5000].shape
# %%
fig, ax = plt.subplots()
ax.set_ylim(25,35)
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
ax.set_title("内挿評価")

name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
for i in [1,500,""]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk{i}_inter_{name}.csv')[::10]
    ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([1,"1/500","1/1000"])
plt.show()
fig.savefig(f"./修論用画像/hyper_nk_inter.png",dpi=300,bbox_inches='tight')

# %%
fig, ax = plt.subplots()
ax.set_ylim(15,30)
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
ax.set_title("外挿評価")

name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
for i in [1,500,""]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk{i}_extra_{name}_0.csv')[::10]
    ax.plot(lc['epoch'],lc['test_loss'])
ax.legend([1,"1/500","1/1000"])

plt.show()
fig.savefig(f"./修論用画像/hyper_nk_extra.png",dpi=300,bbox_inches='tight')

#%%
fig, ax = plt.subplots()
ax.set_ylim(20,35)
ax.set_title("内挿評価と外挿評価の平均")
ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
for i in [1,500,""]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc_i=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk{i}_inter_{name}.csv')[::10]
    lc_e=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk{i}_extra_{name}_0.csv')[::10]
    lc = (lc_i.test_loss+lc_e.test_loss)/2
    ax.plot(lc_i['epoch'],lc)
ax.legend([1,"1/500","1/1000"])
fig.savefig(f"./修論用画像/hyper_nk_interextra.png",dpi=300,bbox_inches='tight')

plt.show()
#%%
fig, ax = plt.subplots(figsize=(20,8))
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")

ax.set_title("主要国における地熱資源量",fontsize=35)
        
# ax.set_xlabel('国名')
ax.set_ylabel('地熱資源量(万kW)',fontsize=30)

bar=ax.bar(['アメリカ\n合衆国','インドネシア','日本','ケニア',"フィリピン","メキシコ","アイスランド","エチオピア","ニュージー\nランド","イタリア","ペルー"],
           [3000,2779,2347,700,600,600,580,500,365,327,300])
bar[2].set_color("red")
fig.savefig(f"./修論用画像/地熱資源量.png",dpi=300,bbox_inches = 'tight')

# %%
grid=pd.read_csv("./input_japan/useful_data/est_grid_detail_ja.csv")
grid.groupby(["x","y"],as_index=False).mean().shape[0]*25
# %%
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data.h_z=input_data.h_z.round(-1)
input_data=input_data.groupby(["x","y","h_z"],as_index=False).mean()

# %%

#%%
parameters = {
    "depth0": [{'sill': 2500-500, 'range': 6000, 'nugget': 500}, 'spherical'],
    "depth500": [{'sill': 3000-500, 'range': 5000, 'nugget': 500}, "spherical"],
    "depth800": [{'sill': 2000-250, 'range': 3000, 'nugget': 250}, 'spherical'],
    "depth1000": [{'sill': 3000-1250, 'range': 20000, 'nugget': 1250}, 'gaussian'],#'sill': 3000-1250, 'range': 20000, 'nugget': 1250
}

def variogram(df, sep=11000, max_dist=500001, parameters={'sill': 9, 'range': 200000, 'nugget': 0.5}, how='gaussian'):
    xy_dis = squareform(pdist(df.iloc[:, [0, 1]]))
    t_vario = squareform(pdist(df.iloc[:, 2].values.reshape(-1, 1))**2)
    sv_i = np.zeros(len(range(0, max_dist, sep)))
    sill_, range_, nugget_ = parameters['sill'], parameters['range'], parameters['nugget']
    for i, value in enumerate(tqdm(range(0, max_dist, sep))):
        mask1 = xy_dis > value
        mask2 = xy_dis < value+sep
        mask = mask1*mask2
        res1 = t_vario[mask]
        mask3 = res1 > 0
        res2 = (res1[mask3].mean())/2
        sv_i[i] = res2
    x = range(0, max_dist, sep)
    plt.plot(x[:], sv_i[:], c='black', marker='o')
    if how == 'gaussian':
        plt.plot(x[:], vm.gaussian_variogram_model(
            [sill_, range_, nugget_], np.array(x))[:], c='red')
    elif how == 'spherical':
        plt.plot(x[:], vm.spherical_variogram_model(
            [sill_, range_, nugget_], np.array(x))[:], c='red')
    print(df.columns[2])
    plt.show()
#%%
a=input_data[input_data.h_z==-500]
a,_=train_test_split(a,train_size=549/767,shuffle=True,random_state=0)
d=squareform(pdist(np.array([a.x.values,a.y.values]).T))
variogram(a[["x","y","t"]], sep=1100, max_dist=20001, parameters={
            'sill': 3000-500, 'range': 5000, 'nugget': 500}, how='spherical')
d_list=[]
for i in range(d.shape[0]):
    d_list.append(np.sort(d[:,i])[1])
np.mean(d_list)
#%%
767
549
245
#%%
# 5619m
# 7592m
# 10666m
# %%
squareform(pdist(np.array([[1,2,3],[4,5,6]]).T))
# %%
np.argmin(d[:,2])
# %%

# %%
fig, ax = plt.subplots()
# ax.grid()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")

ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
ax.set_title("内挿評価")
# plt.rcParams["font.size"] = 20
ax.set_ylim(20,40)
diff_list=[]
for i in ["basic"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    ax.plot(lc['epoch'],lc['test_loss'])
    diff_list.append(lc["test_loss"][10])
#%%
fig, ax = plt.subplots()
# ax.grid()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")

ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
ax.set_title("内挿評価と外挿評価の平均")
# plt.rcParams["font.size"] = 20
ax.set_ylim(15,40)

diff_list=[]
# for i in ["basic","basic_volcano","basic_curie","basic_tishitsu_ohe","basic_onsen","basic_depth","basic_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc_i=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
#     lc_e=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
#     lc=(lc_i.test_loss+lc_e.test_loss)/2
#     ax.plot(lc_i['epoch'],lc)
#     # diff_list.append(lc["test_loss"][10])
for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc_i=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_inter_{i}.csv')
    lc_e=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    lc=(lc_i.test_loss+lc_e.test_loss)/2
    ax.plot(lc_i['epoch'],lc,c="blue")
    
for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc_i=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk500_inter_{i}.csv')[::10]
    lc_e=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk500_extra_{i}_0.csv')[::10]
    lc=(lc_i.test_loss+lc_e.test_loss)/2
    ax.plot(lc_i['epoch'],lc,c="red")
ax.legend(["ALLモデル(DNN)","ALLモデル(NK)"])

# ax.legend(["BASEモデル","活火山モデル","キュリー点深度モデル","地質モデル","温泉モデル","標高別地温モデル","地温勾配モデル","ALLモデル(DNN)","ALLモデル(NK)"],ncol = 4,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
fig.savefig(f"./修論用画像/NK_last.png",dpi=300,bbox_inches='tight')

# %%
# %%
fig, ax = plt.subplots()
# ax.grid()
sns.set()
sns.set(font='Yu Gothic')
sns.set_context("talk")

ax.set_xlabel('Num. of Learning Epochs') 
ax.set_ylabel('RMSE(℃)')
# plt.rcParams["font.size"] = 20
ax.set_ylim(15,40)
# ax.set_ylim(15,60)
ax.set_title("外挿評価")
diff_list=[]
# for i in ["basic"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
#     lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
#     ax.plot(lc['epoch'],lc['test_loss'])
#     diff_list.append(lc["test_loss"][10])

for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{i}_0.csv')
    ax.plot(lc['epoch'],lc['test_loss'],c="black")
    diff_list.append(lc["test_loss"][10])

for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_extra_{i}_0.csv')[::10]
    ax.plot(lc['epoch'],lc['test_loss'],c="blue")
    diff_list.append(lc["test_loss"][100])
    
for i in ["basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"]:#"basic_volcano_curie_onsen_depth800_grad"]:#'basic_onsen_tishitsu_ohe_depth800_grad']:
    lc=pd.read_csv(f'./output_japan_last/learning_curve/nk/lc_nk_svf_1000_extra_{i}_0.csv')[::10]
    ax.plot(lc['epoch'],lc['test_loss'],c="red")
    diff_list.append(lc["test_loss"][100])
    
print(diff_list[0]-diff_list[-1])
ax.legend(["ALLモデル(DNN)","ALLモデル(NK)","ALLモデル(NK_f)"])
# %%