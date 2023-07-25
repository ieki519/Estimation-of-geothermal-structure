
# %%
from numpy.lib.ufunclike import fix
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pykrige.variogram_models as vm

from optuna.samplers import TPESampler
import optuna
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from xgboost.callback import LearningRateScheduler
import time
import datetime
print(datetime.datetime.now())
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(device)
# %%


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


fix_seed(0)
# %%


def preprocess_input(df):
    df['t'] = np.where(df['t'].values <= 0, 0.1, df['t'].values)
    add_input_volcano = pd.read_csv(
        './input_japan/volcano/add_input_volcano_ja.csv')
    add_input_curie = pd.read_csv(
        './input_japan/curie_point/add_input_curie_ja.csv')
    add_input_tishitsu = pd.read_csv(
        './input_japan/tishitsu/add_input_tishitsu_pred_ja.csv')
    add_input_onsen = pd.read_csv('./input_japan/onsen/add_input_onsen_ja.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])

    # df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df = pd.concat([df, tmp], axis=1)

    df = df.drop(['symbol', 'symbol_freq', 'formationAge_ja', 'group_freq',
                  'lithology_ja', 'lithology_freq'], axis=1)  # , 'group_ja'
    df['age'] = (df['age_a']+df['age_b'])/2

    return df


def preprocess_grid(df):
    add_grid_volcano = pd.read_csv(
        './input_japan/volcano/add_grid_volcano_detail_ja.csv')
    add_grid_curie = pd.read_csv(
        './input_japan/curie_point/add_grid_curie_detail_ja.csv')
    add_grid_tishitsu = pd.read_csv(
        './input_japan/tishitsu/add_grid_tishitsu_detail_pred_ja.csv')
    add_grid_onsen = pd.read_csv(
        './input_japan/onsen/add_grid_onsen_detail_ja.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])

    # df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df = pd.concat([df, tmp], axis=1)

    df = df.drop(['symbol', 'symbol_freq', 'formationAge_ja', 'group_freq',
                  'lithology_ja', 'lithology_freq'], axis=1)  # , 'group_ja'
    df['age'] = (df['age_a']+df['age_b'])/2

    return df


# %%
features_dict = {
    "volcano": ['volcano'],
    "curie": ['curie'],
    "onsen": ['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
    "tishitsu_ohe": ['age_a', 'age_b', 'age'],
    "tishitsu_rank": ['age_a', 'age_b', 'age', "group_rank"],
    "depth": ["depth0", "depth500", "depth1000"],  # extraは800
    "grad": ['grad', 'grad_max', 'grad_min', 'grad_max_h_z', 'grad_min_h_z'],
}
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)
master_features = ['x', 'y', 'h', 'z', 'h_z']
# all_f = ["curie","onsen","tishitsu_rank","depth","grad"]
all_f = ["volcano", "curie", "onsen", "tishitsu_ohe", "depth", "grad"]
name = "basic"
for f in all_f:
    master_features += features_dict[f]
curie_data = pd.read_csv('./input_japan/curie_point/grid_curie_ja.csv')
curie_data = preprocess_grid(curie_data)
est_data_detail = pd.read_csv(
    './input_japan/useful_data/est_grid_detail_ja.csv').sort_values(["x", "y", "h_z"])
est_data_detail = preprocess_grid(est_data_detail)


add_input_depth = pd.read_csv(f'./input_japan/depth/add_input_depth_ja.csv')
add_grid_depth = pd.read_csv(
    f'./input_japan/depth/add_grid_depth_detail_ja.csv')

input_data = input_data.merge(add_input_depth, how='left', on=['x', 'y'])
curie_data = curie_data.merge(add_grid_depth, how='left', on=['x', 'y'])
est_data_detail = est_data_detail.merge(
    add_grid_depth, how='left', on=['x', 'y'])

add_input_grad = pd.read_csv(f'./input_japan/grad/add_input_grad_ja.csv')
add_grid_grad = pd.read_csv(f'./input_japan/grad/add_grid_grad_detail_ja.csv')
input_data = input_data.merge(add_input_grad, how='left', on=['x', 'y'])
curie_data = curie_data.merge(add_grid_grad, how='left', on=['x', 'y'])
est_data_detail = est_data_detail.merge(
    add_grid_grad, how='left', on=['x', 'y'])
# %%
name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
# .groupby(["x","y"],as_index=False).mean()[["x","y","t"]]
est_data_t = pd.read_csv(
    f'./output_japan_last/voxler/nk/est_nk500_output_{name}_detail.csv')
est_data = est_data_detail.copy()
est_data = est_data.merge(est_data_t, how="left", on=["x", "y", "h_z"])
est_data
# %%
# est_data_detail=est_data_detail[est_data_detail.z>=0].reset_index(drop=True)
# a=est_data_detail[est_data_detail.h_z==0]
# plt.scatter(a.x,a.y,c="k",s=0.1)
# %%
est_data = est_data_detail.copy()
# input_data = input_data.groupby(["x","y"],as_index=False).mean()
# est_data = est_data.groupby(["x","y"],as_index=False).mean()
# est_data_t=pd.read_csv(f'./output_japan_last/voxler/nk/est_nk500_output_{name}_detail.csv').groupby(["x","y"],as_index=False).mean()[["x","y","t"]]
# est_data=est_data.merge(est_data_t,how="left",on=["x","y"])

est_data = est_data.merge(est_data_t, how="left", on=["x", "y", "h_z"]).groupby(["x","y"],as_index=False).mean()

X_train = input_data[["volcano", "curie", 'age', 'group_ja_堆積岩', 'group_ja_火成岩', "Temp",
                      "Na", "Cl", "grad", "grad_max", "grad_max_h_z", "depth0", "depth500", "depth1000"]].values
Y_train = input_data[["t"]].values

X_est = est_data[["volcano", "curie", 'age', 'group_ja_堆積岩', 'group_ja_火成岩', "Temp", "Na",
                  "Cl", "grad", "grad_max", "grad_max_h_z", "depth0", "depth500", "depth1000"]].values
Y_est = est_data[["t"]].values
scaler = MinMaxScaler()
X_est = scaler.fit_transform(X_est)
X_est_t = Variable(torch.from_numpy(X_est).float()).to(device)
dist_t = torch.sqrt(pairwise_distances(X_est_t))
dist = squareform(pdist(X_est))
#%%
dist[0],dist_t[0]
#%%
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_est = scaler.transform(X_est)
# import pykrige.variogram_models as vm
# X_train
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

def torch_select_column_by_condition(data, index, condition):
    condition_data = data[:, index]
    mask = condition_data.eq(condition)

    if len(torch.nonzero(mask)) == 0:
        return torch.Tensor()
    indices = torch.squeeze(torch.nonzero(mask), 1)
    select = torch.index_select(data, 0, indices)
    return select

def variogram(x_train, y_train, sep=11000, max_dist=500001, parameters={'sill': 9, 'range': 200000, 'nugget': 0.5}, how='gaussian'):
    xy_dis = squareform(pdist(x_train))*100
    t_vario = squareform(pdist(y_train.reshape(-1, 1))**2)
    sv_i = np.zeros(len(range(0, max_dist, sep)))
    sill_, range_, nugget_ = parameters['sill'], parameters['range'], parameters['nugget']
    for i, value in enumerate(range(0, max_dist, sep)):
        mask1 = xy_dis > value
        mask2 = xy_dis <= value+sep
        mask = mask1*mask2
        res1 = t_vario[mask]
        mask3 = res1 > 0
        res2 = (res1[mask3].mean())
        sv_i[i] = res2
    print(sv_i)
    x = range(0, max_dist, sep)
    plt.plot(x[:], sv_i[:], marker='o')  # c='black',
    if how == 'gaussian':
        plt.plot(x[:], vm.gaussian_variogram_model(
            [sill_, range_, nugget_], np.array(x))[:], c='red')
    elif how == 'spherical':
        plt.plot(x[:], vm.spherical_variogram_model(
            [sill_, range_, nugget_], np.array(x))[:], c='red')

    # print(df.columns[2])
    # plt.show()
# %%
variogram(X_train, Y_train, 25, 150, parameters={
          'sill': 2500-250, 'range': 110, 'nugget': 250}, how='ga')
variogram(X_est, Y_est, 25, 150, parameters={
          'sill': 2500-250, 'range': 110, 'nugget': 250}, how='ga')
# %%
scaler = MinMaxScaler()
scaler.fit(input_data[["volcano", "curie", 'age', 'group_ja_堆積岩', 'group_ja_火成岩', "Temp", "Na",
                  "Cl", "grad", "grad_max", "grad_max_h_z", "depth0", "depth500", "depth1000"]].values)
input_data.h_z = input_data.h_z.round(-1)
input_data = input_data.groupby(["x", "y", "h_z"], as_index=False).mean()
h_z_list = np.sort(np.unique(input_data.h_z.values))
legend_list = []
for h_z in [ -1000, -500, 0]:
    df = input_data[input_data.h_z == h_z].reset_index(drop=True)
    df_est = est_data[est_data.h_z == h_z].reset_index(drop=True)
    # if df.shape[0]<400:
    #     continue
    X_train = df[["volcano", "curie", 'age', 'group_ja_堆積岩', 'group_ja_火成岩', "Temp", "Na",
                  "Cl", "grad", "grad_max", "grad_max_h_z", "depth0", "depth500", "depth1000"]].values
    Y_train = df[["t"]].values
    X_est = df_est[["volcano", "curie", 'age', 'group_ja_堆積岩', 'group_ja_火成岩', "Temp", "Na",
                    "Cl", "grad", "grad_max", "grad_max_h_z", "depth0", "depth500", "depth1000"]].values
    Y_est = df_est[["t"]].values
    # print(h_z,df.shape)
    if h_z in [-1000,-500,0]:
        X_train = scaler.transform(X_train)
        variogram(X_train,Y_train,25,150,parameters={'sill': 2500-250, 'range': 110, 'nugget': 250},how='ga')
        legend_list.append(f"{int(h_z)},{df.shape[0]}")
    X_est = scaler.transform(X_est)
    variogram(X_est, Y_est, 25, 150, parameters={
              'sill': 2500-250, 'range': 110, 'nugget': 250}, how='ga')
    legend_list.append(f"est{int(h_z)}")
plt.legend(legend_list, loc="lower right", bbox_to_anchor=(1.3, 0))
# %%
def calc_model_feature_semivariogram(df_input,df_est):
    # preprocess
    sv_feature = ["volcano", "curie", 'age', 'group_ja_堆積岩', 'group_ja_火成岩', "Temp", "Na",
                  "Cl", "grad", "grad_max", "grad_max_h_z", "depth0", "depth500", "depth1000"]
    df_input = df_input.copy()
    df_est = df_est.copy()
    df_input.h_z = df_input.h_z.round(-1)
    df_input = df_input.groupby(["x", "y", "h_z"], as_index=False).mean()
    df_est = df_est.groupby(["x","y"],as_index=False).mean()
    
    # スケーリング
    pc_scaler = MinMaxScaler()
    df_input_xy = df_input.groupby(["x", "y"], as_index=False).mean()
    pc_scaler.fit(df_input_xy[sv_feature].values)
    
    #est準備
    pc_est_f = pc_scaler.transform(df_est[sv_feature].values)
    pc_est_f=Variable(torch.from_numpy(pc_est_f).float()).to(device)
    dist_est_f = torch.sqrt(pairwise_distances(pc_est_f))*100
    
    # calc model sv
    sv_all = []
    for h_z in [-1000,-500,0]:
        # preprocess
        pc_input_f = df_input[df_input.h_z==h_z][sv_feature].values
        pc_input_t = df_input[df_input.h_z==h_z][["t"]].values
        pc_input_f =  pc_scaler.transform(pc_input_f)
        
        dist_input_f = squareform(pdist(pc_input_f))*100
        dist_t = squareform(pdist(pc_input_t.reshape(-1, 1))**2)

        sep,max_dist = 25,101
        sv_i = np.zeros(len(range(0,max_dist,sep)))
        for i , value in enumerate(range(0,max_dist,sep)):
            mask1 = dist_input_f>value
            mask2 = dist_input_f<=value + sep
            mask = mask1*mask2
            res = dist_t[mask]
            if res.size:
                sv_i[i] = res[res>0].mean()
            else:
                sv_i[i] = np.nan
        sv_all.append(sv_i)
    sv_all=np.vstack(sv_all)
    sv_all = Variable(torch.from_numpy(sv_all).float()).to(device)
    
    # calc d xy list
    d_f_list = []
    for i , value in enumerate(range(0,max_dist,sep)):
        mask1 = dist_est_f>value
        mask2 = dist_est_f<=value + sep
        mask = mask1*mask2
        d_f_list.append(mask)
        
    return sv_all,d_f_list
#%%
def calc_pred_feature_semivariogram(ts_est,dist_f_list):
    sv_all=[]
    for h_z in [-1000,-500,0]:
        pc = torch_select_column_by_condition(ts_est,2,h_z)
        pred_t = pc[:,[3]]
        dist_t =pairwise_distances(pred_t)
        sv_i = []
        for i in range(len(dist_f_list)):
            sv_i.append(dist_t[dist_f_list[i]].mean())
        sv_i = torch.stack(sv_i)
        sv_all.append(sv_i)
    sv_all = torch.stack(sv_all)
    return sv_all
#%%
# %%
features_dict = {
    "volcano": ['volcano'],
    "curie": ['curie'],
    "onsen": ['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
    "tishitsu_ohe": ['age_a', 'age_b', 'age'],
    "tishitsu_rank": ['age_a', 'age_b', 'age', "group_rank"],
    "depth": ["depth0", "depth500", "depth1000"],  # extraは800
    "grad": ['grad', 'grad_max', 'grad_min', 'grad_max_h_z', 'grad_min_h_z'],
}
input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
input_data = preprocess_input(input_data)
master_features = ['x', 'y', 'h', 'z', 'h_z']
# all_f = ["curie","onsen","tishitsu_rank","depth","grad"]
all_f = ["volcano", "curie", "onsen", "tishitsu_ohe", "depth", "grad"]
name = "basic"
for f in all_f:
    master_features += features_dict[f]
curie_data = pd.read_csv('./input_japan/curie_point/grid_curie_ja.csv')
curie_data = preprocess_grid(curie_data)
est_data_detail = pd.read_csv(
    './input_japan/useful_data/est_grid_detail_ja.csv').sort_values(["x", "y", "h_z"])
est_data_detail = preprocess_grid(est_data_detail)


add_input_depth = pd.read_csv(f'./input_japan/depth/add_input_depth_ja.csv')
add_grid_depth = pd.read_csv(
    f'./input_japan/depth/add_grid_depth_detail_ja.csv')

input_data = input_data.merge(add_input_depth, how='left', on=['x', 'y'])
curie_data = curie_data.merge(add_grid_depth, how='left', on=['x', 'y'])
est_data_detail = est_data_detail.merge(
    add_grid_depth, how='left', on=['x', 'y'])

add_input_grad = pd.read_csv(f'./input_japan/grad/add_input_grad_ja.csv')
add_grid_grad = pd.read_csv(f'./input_japan/grad/add_grid_grad_detail_ja.csv')
input_data = input_data.merge(add_input_grad, how='left', on=['x', 'y'])
curie_data = curie_data.merge(add_grid_grad, how='left', on=['x', 'y'])
est_data_detail = est_data_detail.merge(
    add_grid_grad, how='left', on=['x', 'y'])
input_data
# %%
name = "basic_volcano_curie_onsen_tishitsu_ohe_depth_grad"
# .groupby(["x","y"],as_index=False).mean()[["x","y","t"]]
est_data_t = pd.read_csv(
    f'./output_japan_last/voxler/nk/est_nk500_output_{name}_detail.csv')
est_data = est_data_detail.copy()
est_data = est_data.merge(est_data_t, how="left", on=["x", "y", "h_z"])
est_data
#%%
pc_est = Variable(torch.from_numpy(est_data[["x","y","h_z","t"]].values).float()).to(device)
#%%
model_sv_f,dist_f_list=calc_model_feature_semivariogram(input_data,est_data)
pred_sv_f = calc_pred_feature_semivariogram(pc_est,dist_f_list)
#%%
torch.sqrt(torch.sqrt(((pred_sv_f-model_sv_f)**2).mean()))
#%%
model_sv_f
# %%
pred_sv_f
# %%
