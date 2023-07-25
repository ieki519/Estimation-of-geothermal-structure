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
import warnings
print(datetime.datetime.now())
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.backends.cudnn.benchmark = True
#%%
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

    df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df=pd.concat([df,tmp],axis=1)
    
    df=df.drop(['symbol','symbol_freq','formationAge_ja', 'group_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    df['age']=(df['age_a']+df['age_b'])/2

    return df

def preprocess_input_est_xy(df):
    # df['t'] = np.where(df['t'].values <= 0, 0.1, df['t'].values)
    add_input_volcano = pd.read_csv('./input_japan/volcano/add_input_volcano_ja.csv')
    add_input_curie = pd.read_csv('./input_japan/curie_point/add_input_curie_ja.csv')
    add_input_tishitsu = pd.read_csv('./input_japan/tishitsu/add_input_tishitsu_pred_ja.csv')
    add_input_onsen = pd.read_csv('./input_japan/onsen/add_input_onsen_ja.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])

    df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df=pd.concat([df,tmp],axis=1)
    
    df=df.drop(['symbol','symbol_freq','formationAge_ja', 'group_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    df['age']=(df['age_a']+df['age_b'])/2

    return df

def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('./input_japan/volcano/add_grid_volcano_detail_ja.csv')
    add_grid_curie = pd.read_csv('./input_japan/curie_point/add_grid_curie_detail_ja.csv')
    add_grid_tishitsu = pd.read_csv('./input_japan/tishitsu/add_grid_tishitsu_detail_pred_ja.csv')
    add_grid_onsen = pd.read_csv('./input_japan/onsen/add_grid_onsen_detail_ja.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])

    df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df=pd.concat([df,tmp],axis=1)
    
    df=df.drop(['symbol','symbol_freq','formationAge_ja', 'group_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    df['age']=(df['age_a']+df['age_b'])/2

    return df

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


def semi_variogram_feature(df_input, df):

    dict_condition = {
        'hokkaidou': [(df_input['x']-df_input['y']) <= 1200000, (df['x']-df['y']) <= 1200000],
        'higashi': [((df_input['x']-df_input['y']) > 1200000) & ((df_input['x']+df_input['y']) >= 5000000), ((df['x']-df['y']) > 1200000) & ((df['x']+df['y']) >= 5000000)],
        'nishi': [((df_input['y']+(2/3)*df_input['x']) > 3050000) & ((df_input['x']+df_input['y']) < 5000000), ((df['y']+(2/3)*df['x']) > 3050000) & ((df['x']+df['y']) < 5000000)],
        'kyusyu': [(df_input['y']+(2/3)*df_input['x']) <= 3050000, (df['y']+(2/3)*df['x']) <= 3050000],
    }

    for key in tqdm(['hokkaidou', 'higashi', 'nishi', 'kyusyu']):
        df_input_condition = dict_condition[key][0]
        df_condition = dict_condition[key][1]

        df_input.loc[df_input_condition,
                     'h_z*'] = df_input.loc[df_input_condition, 'h_z']*(187.7875737003602)
        df.loc[df_condition, 'h_z*'] = df.loc[df_condition, 'h_z'] * \
            (187.7875737003602)
        df_input_xyz = df_input[df_input_condition][['x', 'y', 'h_z*']]
        df_xyz = df[df_condition][['x', 'y', 'h_z*']]

        pair_dist_xyz = cdist(df_xyz.values, df_input_xyz.values)
        t_list = []
        d_list = []
        pair_dist_xyz = np.where(pair_dist_xyz == 0, np.inf, pair_dist_xyz)
        t_values = df_input[df_input_condition]['t'].values
        for i in range(pair_dist_xyz.shape[0]):
            idx_min = np.argmin(pair_dist_xyz[i, :])
            t_list.append(t_values[idx_min])
            d_list.append(pair_dist_xyz[i, idx_min])

        df.loc[df_condition, 'near_t'] = t_list
        df.loc[df_condition, 'near_dist'] = d_list

    del df['h_z*']
    return df


def extra_split(df):
    df = df.sort_values('h_z', ascending=False, ignore_index=True)
    train_data, test_data = train_test_split(df, test_size=0.1, shuffle=False)
    return train_data, test_data


def convert_output(output_data, debug=False):

    ret = output_data

    global log_std, log_mean

    ret = ret*log_std+log_mean

    ret = torch.exp(ret)

    if debug:
        print(ret)

    return ret

# 入力値の変換

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

# def torch_select_column_by_condition_2(data, index, condition):
#     condition_data = data[:, index]
#     mask = condition_data.eq(condition)
#     mask, mask_ = mask.min(1)
#     if len(torch.nonzero(mask)) == 0:
#         return torch.Tensor()
#     indices = torch.squeeze(torch.nonzero(mask), 1)
#     select = torch.index_select(data, 0, indices)
#     return select

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

def calc_model_semivariogram_xy(df_input,df_est):
    model_sv = []
    df_input = df_input.copy()
    df_est = df_est.copy()
    df_input.h_z = df_input.h_z.round(-2)
    df_input=df_input.groupby(["x","y","h_z"],as_index=False).mean()
    
    df_input_xy=df_input.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    df_est_xy=df_est.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    d_bool_list=cdist(df_est_xy.values,df_input_xy.values)
    d_bool_list = d_bool_list<=100000
    for i in tqdm(range(df_est_xy.shape[0])):
        input_list=[]
        input_xy = df_input_xy[d_bool_list[i,:]].values
        for x,y in input_xy:
            input_list.append(df_input[(df_input.x==x) & (df_input.y==y)])
        input_list = pd.concat(input_list).reset_index(drop=True)
        sv_all=[]
        for h_z in input_list.h_z.unique()[input_list.h_z.unique()<=0]:#
            pc_xy = input_list[input_list.h_z==h_z][["x","y"]].values
            pc_t = input_list[input_list.h_z==h_z][["t"]].values
            dist_xy = cdist(pc_xy,pc_xy)
            dist_t = cdist(pc_t,pc_t)**2
            sep=20000
            max_dist=70001
            sv_i = []
            for value in range(sep//2,max_dist,sep):
                res = dist_t[(dist_xy>=value) & (dist_xy<value+sep)]
                if res.size:
                    res = res.mean()
                else:
                    res = np.nan
                sv_i.append(res)
            sv_all.append(sv_i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sv_all =np.nanmean(np.vstack(sv_all),axis=0)
        model_sv.append(sv_all)
    model_sv = np.vstack(model_sv)
    model_sv = Variable(torch.from_numpy(model_sv).float()).to(device)
    return model_sv

def calc_pred_semivariogram_xy(ts_est,dist_xy_list,sv_value):
    sv_master = []
    # for value in range(0,-5001,-1000):
    for value in [-random.randint(0,5)*1000]:
        pc = torch_select_column_by_condition(ts_est,2,value)
        pred_t = pc[:,[3]]
        dist_t_list = pairwise_distances(pred_t)
        sv_all = []
        for i in range(dist_xy_list.shape[0]):
            sep=20000
            max_dist=70001
            sv_i = []
            # for value in range(sep//2,max_dist,sep):
                # value = sv_value*sep + sep/2
            for value in range(sv_value*sep + sep//2,max_dist,sep*2):
                res = dist_t_list[i,:][(dist_xy_list[i,:]>= value) & (dist_xy_list[i,:] < value+sep)]
                res = res.mean()
                sv_i.append(res)
            sv_i = torch.stack(sv_i)
            sv_all.append(sv_i)
        sv_all=torch.stack(sv_all)
        sv_master.append(sv_all)
    sv_master = torch.stack(sv_master).mean(axis=0)
    return sv_master

# def calc_model_semivariogram_z(df_input,df_est):
#     df_input = df_input.copy()
#     df_est = df_est.copy()
#     df_input.h_z = df_input.h_z.round(-2)
#     df_input_xy=df_input.groupby(["x","y"],as_index=False).mean()[["x","y"]]
#     df_est_xy=df_est.groupby(["x","y"],as_index=False).mean()[["x","y"]]
#     sv_all=[]
#     for x,y in df_input_xy.values:
#         pc=df_input[(df_input.x==x)&(df_input.y==y)]
#         h_z=pc[["h_z"]].values
#         t=pc[["t"]].values
#         dist_z=squareform(pdist(h_z))
#         dist_t=squareform(pdist(t))**2
#         sv_i=np.zeros(len(range(500,1501,500)))
#         for i,dist in enumerate(range(500,1501,500)):
#             mask=dist_z==dist
#             res=dist_t[mask]
#             if res.size:
#                 sv_i[i]=res.mean()
#             else:
#                 sv_i[i]=np.nan
#         sv_all.append(sv_i)
#     sv_all=np.vstack(sv_all)
#     dist_nd=cdist(df_est_xy,df_input_xy)
    
#     model_sv_list=[]
#     for i in range(dist_nd.shape[0]):
#         dist_nd_i = dist_nd[i,:][dist_nd[i,:]<=20000]
#         mask=np.argsort(dist_nd_i)[:5]
#         model_sv_list.append(np.nanmean(sv_all[mask,:],axis=0))
#     model_sv=np.vstack(model_sv_list)
#     model_sv = Variable(torch.from_numpy(model_sv).float()).to(device)
#     return model_sv

def calc_model_semivariogram_z(df_input):
    df_input = df_input.copy()
    # df_input.h_z = df_input.h_z.round(-2)
    df_input_xy=df_input.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    sv_all=[]
    for x,y in df_input_xy.values:
        pc=df_input[(df_input.x==x)&(df_input.y==y)]
        h_z=pc[["h_z"]].values
        t=pc[["t"]].values
        dist_z=squareform(pdist(h_z))
        dist_t=squareform(pdist(t))**2
        sv_i=np.zeros(len(range(500,1001,500)))#1501
        for i,dist in enumerate(range(500,1001,500)):#1501
            # mask=dist_z==dist
            mask1 = dist_z>=dist-50
            mask2 = dist_z<=dist+50
            res=dist_t[mask1*mask2]
            if res.size:
                sv_i[i]=res.mean()
            else:
                sv_i[i]=np.nan
        sv_all.append(sv_i)
    sv_all=np.vstack(sv_all)
    sv_all=pd.DataFrame(sv_all,columns=["sv_500","sv_1000"])
    df_input_xy = pd.concat([df_input_xy,sv_all],axis=1)
    return df_input_xy

def create_d_z_list(ts_est,ts_est_xy):
    dist_z_list=[]
    for xy in ts_est_xy:
        pc=torch_select_column_by_condition_2(ts_est,[0,1],xy)
        h_z=pc[:,[2]]
        dist_z=torch.sqrt(pairwise_distances(h_z))
        dist_z_list.append(dist_z)
    return dist_z_list

# def create_d_z_list(nd_est,nd_est_xy):
#     dist_z_list=[]
#     for x,y in nd_est_xy:
#         pc=nd_est[(nd_est[:,0]==x) & (nd_est[:,1]==y)]
#         h_z=pc[:,[2]]
#         dist_z=squareform(pdist(h_z))
#         dist_z_list.append(dist_z)
#     return dist_z_list

def calc_pred_semivariogram_z(ts_est,ts_est_xy,dist_z_list):
    sv_all=[]
    for i,xy in enumerate(ts_est_xy):
        pc=torch_select_column_by_condition_2(ts_est,[0,1],xy)
        pred_t=pc[:,[3]]
        dist_z=dist_z_list[i]
        dist_t=pairwise_distances(pred_t)
        sv_i=[]
        for z in range(500,1001,500):#1501
            sv_i.append(dist_t[dist_z==z].mean())
        sv_i=torch.stack(sv_i)
        sv_all.append(sv_i)
    sv_all=torch.stack(sv_all)
    return sv_all

def calc_pred_semivariogram_z_2(ts_est,ts_est_xy,dist_z_list):
    sv_all=[]
    for i,xy in enumerate(ts_est_xy):
        pc=torch_select_column_by_condition_2(ts_est,[0,1],xy)
        pred_t=pc[:,[3]]
        dist_z=dist_z_list[i]
        dist_t=pairwise_distances(pred_t)
        sv_i=[]
        for z in range(500,1001,500):#1501
            sv_i.append(dist_t[(dist_z>=z-50)&(dist_z<=z+50)].mean())
        sv_i=torch.stack(sv_i)
        sv_all.append(sv_i)
    sv_all=torch.stack(sv_all)
    return sv_all

def convert_data(input_data, debug=False):
    ret = convert_output(model(input_data), debug)
    return ret

def Min_t(output, omin):
    min_t = ((output - omin).clamp(max=0) ** 2).mean()
    return min_t

def Max_t(output, omax):
    max_t = ((output - omax).clamp(min=0) ** 2).mean()
    return max_t

def Min_t_580(output):
    min_t = ((output - 580).clamp(max=0) ** 2).mean()
    return min_t

def Max_t_580(output):
    max_t = ((output - 580).clamp(min=0) ** 2).mean()
    return max_t

def Stacking_train_model(df_train_s,target_s):
    trn_s=df_train_s.drop(target_s,axis=1).reset_index(drop=True)
    tst_s=df_train_s[target_s].reset_index(drop=True)
    train_features_s=pd.DataFrame(np.zeros((trn_s.shape[0],3)))
    cv_s = KFold(n_splits=5, shuffle=True, random_state=0)
    for trn_idx_s, val_idx_s in cv_s.split(trn_s, tst_s):
        trn_x_s = trn_s.iloc[trn_idx_s, :]
        trn_y_s = tst_s[trn_idx_s]
        val_x_s = trn_s.iloc[val_idx_s, :]
        val_y_s = tst_s[val_idx_s]

        rf=RandomForestRegressor(random_state=0)
        lgbm=LGBMRegressor(random_state=0)
        xgb=XGBRegressor(random_state=0)

        rf.fit(trn_x_s, trn_y_s)
        lgbm.fit(trn_x_s, trn_y_s)
        xgb.fit(trn_x_s, trn_y_s)

        train_features_s.iloc[val_idx_s, 0]=rf.predict(val_x_s)
        train_features_s.iloc[val_idx_s, 1]=lgbm.predict(val_x_s)
        train_features_s.iloc[val_idx_s, 2]=xgb.predict(val_x_s)

    lr=LinearRegression()
    lr.fit(train_features_s,tst_s)
    return lr

def Stacking_est(model_s,df_train_s,target_s,df_est_s):
    trn_s=df_train_s.drop(target_s,axis=1)
    tst_s=df_train_s[target_s]

    rf=RandomForestRegressor(random_state=0)
    lgbm=LGBMRegressor(random_state=0)
    xgb=XGBRegressor(random_state=0)

    rf.fit(trn_s, tst_s)
    lgbm.fit(trn_s, tst_s)
    xgb.fit(trn_s, tst_s)

    est_features_s=pd.DataFrame(np.zeros((df_est_s.shape[0],3)))
    est_features_s[0]=rf.predict(df_est_s)
    est_features_s[1]=lgbm.predict(df_est_s)
    est_features_s[2]=xgb.predict(df_est_s)

    pred_s=model_s.predict(est_features_s)
    return pred_s

class Net(nn.Module):
    def __init__(self, input_size, output_size, unit_size):
        super(Net, self).__init__()
        self.fc_s = nn.Linear(input_size, unit_size)
        self.fc2 = nn.Linear(int(unit_size), int(unit_size))
        self.fc3 = nn.Linear(int(unit_size), int(unit_size))
        self.fc_e = nn.Linear(int(unit_size), output_size)

    def forward(self, x):

        x = F.relu(self.fc_s(x))
        # x=F.dropout(x,p=0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc3(x))
        # x=F.dropout(x,p=0.2)
        x = self.fc_e(x)

        return x
#%%
import argparse
parser = argparse.ArgumentParser() 
parser.add_argument("--number", type=int)
args = parser.parse_args() 
print(args.number)
#%%
features_dict = {
                "volcano":['volcano'],
                "curie":['curie'],
                "onsen":['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
                "tishitsu_ohe":['age_a', 'age_b','age'],
                "tishitsu_rank":['age_a', 'age_b','age',"group_rank"],
                "depth800":[],#extraは1000
                "grad":[],
                }
#%%
# 正規化による出力のズレに気を付ける．
#%%
pred_est_list=[]
learning_curve_list_master=[]
for num in range(5):
    if num==args.number:
        fix_seed(num)
        print(f"output{num}")
        # master_features=['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
        #                     'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
        #                      'group_rank', 'age']
        master_features=['x', 'y', 'h', 'z', 'h_z']
        
        all_f = ["volcano","curie","onsen","tishitsu_rank","depth800","grad"]
        
        name = "basic"
        for f in all_f:
            master_features+=features_dict[f]
            name += "_"+f 
        print(name)
        cat_features = ['group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩']
        
        # preprocess
        input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
        input_data = preprocess_input(input_data)

        train_data, test_data = extra_split(input_data)#0.01にする
        # train_data = input_data.copy()

        est_data = pd.read_csv('./input_japan/useful_data/est_grid_500_ja.csv').sort_values(["x","y","h_z"])
        est_data_detail = pd.read_csv('./input_japan/useful_data/est_grid_detail_ja.csv').sort_values(["x","y","h_z"])
        est_data_input_xy = pd.read_csv('./input_japan/useful_data/est_grid_input_xy_ja.csv').sort_values(["x","y","h_z"])
        curie_data = pd.read_csv('./input_japan/curie_point/grid_curie_ja.csv')
        curie_data_580ika = pd.read_csv('./input_japan/curie_point/grid_curie_580ika_ja.csv')
        curie_data_580izyou = pd.read_csv('./input_japan/curie_point/grid_curie_580izyou_ja.csv')

        est_data = preprocess_grid(est_data)
        est_data_detail = preprocess_grid(est_data_detail)
        est_data_input_xy = preprocess_input_est_xy(est_data_input_xy)
        curie_data = preprocess_grid(curie_data)
        curie_data_580ika = preprocess_grid(curie_data_580ika)
        curie_data_580izyou = preprocess_grid(curie_data_580izyou)
        
        # depth feature 
        if "depth" in name:
            depth_target_list=[0,500,800]# 実際は1000
            depth_features = master_features.copy()
            if "tishitsu_ohe" in name:
                depth_features+=cat_features
            
            train_data_plane = train_data.groupby(['x', 'y'], as_index=False).mean()
            test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
            est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()
            est_data_plane_detail = est_data_detail.groupby(['x', 'y'], as_index=False).mean()
            
            for depth_target in depth_target_list:
                train_data_depth = train_data[train_data.h_z==-depth_target].copy()
                
                depth_target=f"depth{depth_target}"
                train_data_depth[depth_target] =train_data_depth["t"]
                
                X_depth = train_data_depth[depth_features+[depth_target]]
                X_train = train_data_plane[depth_features]
                X_test = test_data_plane[depth_features]
                X_est = est_data_plane[depth_features]
                X_est_detail = est_data_plane_detail[depth_features]

                stacking_model=Stacking_train_model(X_depth,depth_target)
                train_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_train)
                test_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_test)
                est_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_est)
                est_data_plane_detail[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_est_detail)
                
                train_data = train_data.merge(train_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
                test_data = test_data.merge(test_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
                est_data = est_data.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
                est_data_detail = est_data_detail.merge(est_data_plane_detail[['x', 'y', depth_target]], how='left', on=['x', 'y'])
                est_data_input_xy = est_data_input_xy.merge(train_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
                curie_data=curie_data.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
                curie_data_580ika=curie_data_580ika.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
                curie_data_580izyou=curie_data_580izyou.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
                master_features += [depth_target]
                
        if "grad" in name:
            # grad feature
            train_data = grad_calc(train_data)
            train_data=grad_maxmin_calc(train_data)

            train_data_plane = train_data.groupby(
                ['x', 'y'], as_index=False).mean()
            test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
            est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()
            est_data_plane_detail = est_data_detail.groupby(['x', 'y'], as_index=False).mean()

            grad_features = master_features.copy()
            if "tishitsu_ohe" in name:
                    grad_features+=cat_features
            
            grad_target_list = ['grad','grad_max','grad_min','grad_max_h_z','grad_min_h_z']
            
            for grad_target in grad_target_list:
                X_train = train_data_plane[grad_features+[grad_target]]
                X_test = test_data_plane[grad_features]
                X_est = est_data_plane[grad_features]
                X_est_detail = est_data_plane_detail[grad_features]

                stacking_model=Stacking_train_model(X_train,grad_target)
                test_data_plane[grad_target] = Stacking_est(stacking_model,X_train,grad_target,X_test)
                est_data_plane[grad_target] = Stacking_est(stacking_model,X_train,grad_target,X_est)
                est_data_plane_detail[grad_target] = Stacking_est(stacking_model,X_train,grad_target,X_est_detail) 
                
                test_data = test_data.merge(test_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
                est_data = est_data.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
                est_data_detail = est_data_detail.merge(est_data_plane_detail[['x', 'y', grad_target]], how='left', on=['x', 'y']) 
                est_data_input_xy = est_data_input_xy.merge(train_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y']) 
                curie_data=curie_data.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
                curie_data_580ika=curie_data_580ika.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
                curie_data_580izyou=curie_data_580izyou.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
                master_features += [grad_target]
        
        # sv calc 
        # est_data_origin = est_data[['x', 'y', 'h_z']].values
        # est_data_origin_xy = est_data[['x', 'y', 'h_z']].groupby(["x","y"],as_index=False).mean()[["x","y"]].values
        # # to torch
        # est_data_origin = Variable(torch.from_numpy(
        #     est_data_origin).float()).to(device)
        # est_data_origin_xy = Variable(torch.from_numpy(
        #     est_data_origin_xy).double()).to(device)
        
        # # xy
        # # model_sv_xy_origin = calc_model_semivariogram_xy(train_data,est_data)
        # # model_sv_xy_origin = Variable(torch.from_numpy(pd.read_csv("./input/semivariogram/model_sv_extra.csv").values).float()).to(device)
        # # dist_xy_list = torch.sqrt(pairwise_distances(est_data_origin_xy,est_data_origin_xy))
        
        # # z
        # # model_sv_z=calc_model_semivariogram_z(train_data,est_data)
        # dist_z_list = create_d_z_list(est_data_origin,est_data_origin_xy)
        
        # train_sv_plane =calc_model_semivariogram_z(train_data)
        # train_data_plane = train_data.groupby(
        #     ['x', 'y'], as_index=False).mean()
        # train_data_plane = train_data_plane.merge(train_sv_plane,on =["x","y"],how="left")
        # test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
        # est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

        # sv_features = master_features.copy()
        # if "tishitsu_ohe" in name:
        #     sv_features+=cat_features
        
        # sv_target_list = ['sv_500','sv_1000']
        # for sv_target in sv_target_list:
        #     train_data_plane_is = train_data_plane[~train_data_plane[sv_target].isnull()]
        #     train_data_plane_isnull = train_data_plane[train_data_plane[sv_target].isnull()]
        #     train_data_plane_isnull = train_data_plane_isnull.drop(columns=sv_target)

        #     X_train = train_data_plane_is[sv_features+[sv_target]]
        #     X_train_isnull = train_data_plane_isnull[sv_features]
        #     X_test = test_data_plane[sv_features]
        #     X_est = est_data_plane[sv_features]

        #     stacking_model=Stacking_train_model(X_train,sv_target)
        #     train_data_plane_isnull[sv_target] = Stacking_est(stacking_model,X_train,sv_target,X_train_isnull)
        #     test_data_plane[sv_target] = Stacking_est(stacking_model,X_train,sv_target,X_test)
        #     est_data_plane[sv_target] = Stacking_est(stacking_model,X_train,sv_target,X_est)

        #     train_data_plane = pd.concat([train_data_plane_is,train_data_plane_isnull],axis=0)

        #     train_data = train_data.merge(train_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
        #     test_data = test_data.merge(test_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
        #     est_data = est_data.merge(est_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
        #     curie_data=curie_data.merge(est_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
        #     curie_data_580ika=curie_data_580ika.merge(est_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
        #     curie_data_580izyou=curie_data_580izyou.merge(est_data_plane[['x', 'y', sv_target]], how='left', on=['x', 'y'])
        #     # master_features += [sv_target]
        # model_sv_z = est_data.groupby(["x","y"],as_index=False).mean()[sv_target_list].values
        # model_sv_z = Variable(torch.from_numpy(model_sv_z).float()).to(device)
        
        # main
        features = master_features.copy()
        print(features)
        target = 't'

        X_train = train_data[features]
        Y_train = train_data[target]

        X_test = test_data[features]
        Y_test = test_data[target]

        X_est = est_data[features]
        X_est_detail = est_data_detail[features]
        X_est_input_xy = est_data_input_xy[features]

        X_curie = curie_data[features]
        Y_curie = curie_data[target]

        X_curie_580ika=curie_data_580ika[features]
        X_curie_580izyou=curie_data_580izyou[features]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_est = scaler.transform(X_est)
        X_est_detail = scaler.transform(X_est_detail)
        X_est_input_xy = scaler.transform(X_est_input_xy)
        X_curie = scaler.transform(X_curie)
        X_curie_580ika=scaler.transform(X_curie_580ika)
        X_curie_580izyou=scaler.transform(X_curie_580izyou)
        
        if "tishitsu_ohe" in name:
            print(cat_features)
            X_train = np.hstack((X_train,train_data[cat_features].values))
            X_test = np.hstack((X_test,test_data[cat_features].values))
            X_est = np.hstack((X_est,est_data[cat_features].values))
            X_est_detail = np.hstack((X_est_detail,est_data_detail[cat_features].values))
            X_est_input_xy = np.hstack((X_est_input_xy,est_data_input_xy[cat_features].values))
            X_curie = np.hstack((X_curie,curie_data[cat_features].values))
            X_curie_580ika = np.hstack((X_curie_580ika,curie_data_580ika[cat_features].values))
            X_curie_580izyou = np.hstack((X_curie_580izyou,curie_data_580izyou[cat_features].values))

        log_target = np.log(Y_train)
        log_mean = log_target.mean()
        log_std = log_target.std()
        omin = 0

        X_train = Variable(torch.from_numpy(X_train).float()).to(device)
        Y_train = Variable(torch.from_numpy(
            Y_train.values.reshape(-1, 1)).float()).to(device)
        X_test = Variable(torch.from_numpy(X_test).float()).to(device)
        Y_test = Variable(torch.from_numpy(
            Y_test.values.reshape(-1, 1)).float()).to(device)
        X_est = Variable(torch.from_numpy(X_est).float()).to(device)
        X_est_detail = Variable(torch.from_numpy(X_est_detail).float()).to(device)
        X_est_input_xy = Variable(torch.from_numpy(X_est_input_xy).float()).to(device)
        X_curie = Variable(torch.from_numpy(X_curie).float()).to(device)
        Y_curie = Variable(torch.from_numpy(
            Y_curie.values.reshape(-1, 1)).float()).to(device)
        X_curie_580ika = Variable(torch.from_numpy(X_curie_580ika).float()).to(device)
        X_curie_580izyou = Variable(torch.from_numpy(X_curie_580izyou).float()).to(device)

        loss_list = []
        learning_curve_list=[]
        model = Net(X_train.shape[1], 1, 150).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        model.load_state_dict(torch.load(f'./output_japan/model/torch_nk_extra_{name}_{num}.pth'))

        time_start = time.time()

        model.eval()
        # pred_est_list.append(convert_data(X_est).detach().cpu().numpy())
        # learning_curve_list_master.append(np.array(learning_curve_list))
        
        #save
        print('-'*50)
        # est_data['t'] = convert_data(X_est).detach().cpu().numpy()
        # est_data = est_data[['x', 'y', 'h_z', 't']]
        # est_data.to_csv(f'./output_japan/voxler/nk/est_nk_output_{name}_{num}.csv', index=False)
        
        
        # est_data_detail['t'] = convert_data(X_est_detail).detach().cpu().numpy()
        # est_data_detail = est_data_detail[['x', 'y', 'h_z', 't']]
        # est_data_detail.to_csv(f'./output_japan/voxler/nk/est_nk_output_{name}_{num}_detail.csv', index=False)
        
        # est_data_detail['t'] = convert_data(X_est_detail).detach().cpu().numpy()
        # est_data_detail = est_data_detail[['x', 'y', 'h_z', 't']]
        # est_data_detail.to_csv(f'./output_japan/voxler/nk/est_nk_output_{name}_{num}_detail.csv', index=False)
        
        est_data_input_xy['t'] = convert_data(X_est_input_xy).detach().cpu().numpy()
        est_data_input_xy = est_data_input_xy[['x', 'y', 'h_z', 't']]
        est_data_input_xy.to_csv(f'./output_japan/voxler/est_nk_extra_{name}_{num}_input_xy.csv', index=False)
        print((time.time()-time_start)/60)
# %%
