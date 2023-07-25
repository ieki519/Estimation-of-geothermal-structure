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
print(datetime.datetime.now())
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    add_input_volcano = pd.read_csv('./input/volcano/add_input_volcano.csv')
    add_input_curie = pd.read_csv('./input/curie_point/add_input_curie.csv')
    add_input_tishitsu = pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
    add_input_onsen = pd.read_csv('./input/onsen/add_input_onsen.csv')
    add_input_kmeans = pd.read_csv('./input/k_means/add_input_kmeans.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])
    df = df.merge(add_input_kmeans, how='left', on=['x', 'y'])

    df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    df=df.drop(['symbol','symbol_freq','formationAge_ja', 'group_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    df['age']=(df['age_a']+df['age_b'])/2

    return df

def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('./input/volcano/add_grid_volcano.csv')
    add_grid_curie = pd.read_csv('./input/curie_point/add_grid_curie.csv')
    add_grid_tishitsu = pd.read_csv('./input/tishitsu/add_grid_tishitsu_pred.csv')
    add_grid_onsen = pd.read_csv('./input/onsen/add_grid_onsen.csv')
    add_grid_kmeans = pd.read_csv('./input/k_means/add_grid_kmeans.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])
    df = df.merge(add_grid_kmeans, how='left', on=['x', 'y'])

    df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
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
        if zt.shape[0] == 1:
            zt = pd.DataFrame(np.array([[0, 0]]), columns=[
                              'z', 't']).append(zt)
        zt = zt.sort_values('z')
        z_diff = np.diff(zt['z'])
        t_diff = np.diff(zt['t'])
        grad = (t_diff/z_diff)*1000
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_max'] = max(grad)
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad_min'] = min(grad)
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

def create_d_list(est_data, z_unique, max_d):
    d_list = []
    bool_list = []
    for i in z_unique:
        pc = torch_select_column_by_condition(est_data, 2, i).double()
        pc = torch.sqrt(pairwise_distances(pc[:, [0, 1]]))
        bool_list_sub = []
        for j in range(pc.shape[0]):
            mask1 = pc[j, :] <= max_d
            mask2 = pc[j, :] > 0
            d_list.append(pc[j, :][mask1*mask2])
            bool_list_sub.append(mask1*mask2)
        bool_list.append(bool_list_sub)
    d_list = torch.cat(d_list)
    return d_list, bool_list


def create_t_list(pred_est_data, z_unique, bool_list):
    t_list = []
    for i, value in enumerate(z_unique):
        pc = torch_select_column_by_condition(pred_est_data, 2, value)
        for j in range(pc.shape[0]):
            vario_t = ((pc[j, 3]-pc[:, 3][[bool_list[i][j]]])**2)/2
            t_list.append(vario_t)
    t_list = torch.cat(t_list)
    return t_list


def region_condition(df):
    dict_condition = {
        'hokkaidou': [(df['x']-df['y']) <= 1200000, 40001],
        'higashi': [((df['x']-df['y']) > 1200000) & ((df['x']+df['y']) >= 5000000), 80001],
        'nishi': [((df['y']+(2/3)*df['x']) > 3050000) & ((df['x']+df['y']) < 5000000)],
        'kyusyu': [(df['y']+(2/3)*df['x']) <= 3050000, 80001],
    }
    return dict_condition

def region_condition_torch(ts):
    dict_condition = {
        'hokkaidou': [(ts[:, 0]-ts[:, 1]) <= 1200000, 40001],
        'higashi': [((ts[:, 0]-ts[:, 1]) > 1200000) & ((ts[:, 0]+ts[:, 1]) >= 5000000), 80001],
        'nishi': [((ts[:, 1]+(2/3)*ts[:, 0]) > 3050000) & ((ts[:, 0]+ts[:, 1]) < 5000000)],
        'kyusyu': [(ts[:, 1]+(2/3)*ts[:, 0]) <= 3050000, 80001],
    }
    return dict_condition


def calc_model_semivariogram_xy(df_input, region):
    features = ['x', 'y', 'h_z', 't']
    df_input = df_input[features]
    dict_conditon = region_condition(df_input)
    df_input = df_input[dict_conditon[region][0]]
    df_input['h_z'] = df_input['h_z'].round(-2)
    df_input = df_input.groupby(['x', 'y', 'h_z'], as_index=False).mean()
    z_unique = df_input['h_z'].unique()
    sv_all = []
    for z in z_unique:
        pc = df_input[df_input['h_z'] == z]
        pc_xy_dis = squareform(pdist(pc[['x', 'y']]))
        pc_t_vario = squareform((pdist(pc['t'].values.reshape(-1, 1))**2)/2)
        sep = 11000
        max_d = dict_conditon[region][1]
        sv_i = np.zeros(len(range(0, max_d, sep)))
        for i, value in enumerate(range(0, max_d, sep)):
            mask1 = pc_xy_dis > value
            mask2 = pc_xy_dis < value+sep
            mask = mask1*mask2
            res1 = pc_t_vario[mask]
            mask3 = res1 > 0
            res2 = res1[mask3]
            if res2.size:
                sv_i[i] = res2.mean()
            else:
                sv_i[i] = np.nan
        sv_all.append(sv_i)
    sv_all = np.vstack(sv_all)
    result = []
    for i in range(len(range(0, max_d, sep))):
        result.append(sv_all[:, i][~np.isnan(sv_all[:, i])].mean())
    return result

def create_d_bool_dict(ts):
    dict_conditon = region_condition_torch(ts)
    d_bool_dict = dict()
    for region in ['hokkaidou', 'higashi', 'kyusyu']:
        pc = ts[dict_conditon[region][0]]
        z = torch.unique(pc[:, 2])
        max_d = dict_conditon[region][1]
        d_list, bool_list = create_d_list(pc, z, max_d)
        d_bool_dict[region] = [d_list, bool_list]
    return d_bool_dict

def calc_pred_semivariogram_xy(ts, d_bool_dict, z_unique, region):
    dict_conditon = region_condition_torch(ts)
    ts = ts[dict_conditon[region][0]]
    sep = 11000
    max_d = dict_conditon[region][1]
    d_list, bool_list = d_bool_dict[region]
    t_list = create_t_list(ts, z_unique, bool_list)
    sv_list = []
    for i, value in enumerate(range(0, max_d, sep)):
        mask1 = d_list > value
        mask2 = d_list < value+sep
        mask = mask1*mask2
        res1 = t_list[mask]
        mask3 = res1 > 0
        res2 = (res1[mask3].mean())
        sv_list.append(res2)
    sv = torch.stack(sv_list)
    return sv

def calc_model_semivariogram_z(df_input,df_est):
    df_input = df_input.copy()
    df_est = df_est.copy()
    df_input.h_z = df_input.h_z.round(-2)
    df_input_xy=df_input.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    df_est_xy=df_est.groupby(["x","y"],as_index=False).mean()[["x","y"]]
    sv_all=[]
    for x,y in df_input_xy.values:
        pc=df_input[(df_input.x==x)&(df_input.y==y)]
        h_z=pc[["h_z"]].values
        t=pc[["t"]].values
        dist_z=squareform(pdist(h_z))
        dist_t=squareform(pdist(t))
        sv_i=np.zeros(len(range(500,1501,500)))
        for i,dist in enumerate(range(500,1501,500)):
            mask=dist_z==dist
            res=dist_t[mask]
            if res.size:
                sv_i[i]=res.mean()
            else:
                sv_i[i]=np.nan
        sv_all.append(sv_i)
    sv_all=np.vstack(sv_all)
    dist_nd=cdist(df_est_xy,df_input_xy)
    
    model_sv_list=[]
    for i in range(dist_nd.shape[0]):
        mask=dist_nd[i,:]<=80000
        model_sv_list.append(np.nanmean(sv_all[mask,:],axis=0))
    model_sv=np.vstack(model_sv_list)
    model_sv = Variable(torch.from_numpy(model_sv).float()).to(device)
    return model_sv

def create_d_z_list(ts_est,ts_est_xy):
    dist_z_list=[]
    for xy in ts_est_xy:
        pc=torch_select_column_by_condition_2(ts_est,[0,1],xy)
        h_z=pc[:,[2]]
        dist_z=torch.sqrt(pairwise_distances(h_z))
        dist_z_list.append(dist_z)
    return dist_z_list

def calc_pred_semivariogram_z(ts_est,ts_est_xy,dist_z_list):
    sv_all=[]
    for i,xy in enumerate(ts_est_xy):
        pc=torch_select_column_by_condition_2(ts_est,[0,1],xy)
        pred_t=pc[:,[3]]
        dist_z=dist_z_list[i]
        dist_t=torch.sqrt(pairwise_distances(pred_t))
        sv_i=[]
        for z in range(500,1501,500):
            sv_i.append(dist_t[dist_z==z].mean())
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
pred_est_list=[]
learning_curve_list_master=[]
for num in range(5):
    if num==4:
        fix_seed(num)
        print(num)
        master_features=['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
                            'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
                             'group_rank', 'age']
        
        # preprocess
        input_data = pd.read_csv('./input/useful_data/input_data.csv')
        input_data = preprocess_input(input_data)

        train_data, test_data = extra_split(input_data)

        est_data = pd.read_csv('./input/useful_data/est_grid_500.csv')
        curie_data = pd.read_csv('./input/curie_point/grid_curie.csv')
        curie_data_580ika = pd.read_csv('./input/curie_point/grid_curie_580ika.csv')
        curie_data_580izyou = pd.read_csv('./input/curie_point/grid_curie_580izyou.csv')

        est_data = preprocess_grid(est_data)
        curie_data = preprocess_grid(curie_data)
        curie_data_580ika = preprocess_grid(curie_data_580ika)
        curie_data_580izyou = preprocess_grid(curie_data_580izyou)

        # sv calc 
        # xy
        model_sv_dict = {}
        for region in ['hokkaidou', 'higashi', 'kyusyu']:
            model_sv = np.array(
                calc_model_semivariogram_xy(train_data, region))
            model_sv_dict[region] = Variable(
                torch.from_numpy(model_sv).float()).to(device)
        est_data_origin = est_data[['x', 'y', 'h_z']].values
        est_data_origin_xy = est_data[['x', 'y', 'h_z']].groupby(["x","y"],as_index=False).mean()[["x","y"]].values
        
        est_data_origin = Variable(torch.from_numpy(
            est_data_origin).float()).to(device)
        est_data_origin_xy = Variable(torch.from_numpy(
            est_data_origin_xy).float()).to(device)
        
        z_unique = torch.unique(est_data_origin[:, 2])
        d_bool_dict = create_d_bool_dict(est_data_origin)
        # z
        model_sv_z=calc_model_semivariogram_z(train_data,est_data)
        dist_z_list = create_d_z_list(est_data_origin,est_data_origin_xy)
        
        # depth feature 
        depth_target_list=[0,500,800]
        depth_features = master_features.copy()
        
        train_data_plane = train_data.groupby(['x', 'y'], as_index=False).mean()
        test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
        est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()
        
        for depth_target in depth_target_list:
            train_data_depth = train_data[train_data.h_z==-depth_target].copy()
            
            depth_target=f"depth{depth_target}"
            train_data_depth[depth_target] =train_data_depth["t"]
            
            X_depth = train_data_depth[depth_features+[depth_target]]
            X_train = train_data_plane[depth_features]
            X_test = test_data_plane[depth_features]
            X_est = est_data_plane[depth_features]

            stacking_model=Stacking_train_model(X_depth,depth_target)
            train_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_train)
            test_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_test)
            est_data_plane[depth_target] = Stacking_est(stacking_model,X_depth,depth_target,X_est)
            
            train_data = train_data.merge(train_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
            test_data = test_data.merge(test_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
            est_data = est_data.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
            curie_data=curie_data.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
            curie_data_580ika=curie_data_580ika.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
            curie_data_580izyou=curie_data_580izyou.merge(est_data_plane[['x', 'y', depth_target]], how='left', on=['x', 'y'])
            master_features += [depth_target]

        # grad feature
        train_data = grad_calc(train_data)
        train_data=grad_maxmin_calc(train_data)

        train_data_plane = train_data.groupby(
            ['x', 'y'], as_index=False).mean()
        test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
        est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

        grad_features = master_features.copy()
        grad_target_list = ['grad','grad_max','grad_min']

        for grad_target in grad_target_list:
            X_train = train_data_plane[grad_features+[grad_target]]
            X_test = test_data_plane[grad_features]
            X_est = est_data_plane[grad_features]

            stacking_model=Stacking_train_model(X_train,grad_target)
            test_data_plane[grad_target] = Stacking_est(stacking_model,X_train,grad_target,X_test)
            est_data_plane[grad_target] = Stacking_est(stacking_model,X_train,grad_target,X_est)
            
            test_data = test_data.merge(test_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
            est_data = est_data.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
            curie_data=curie_data.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
            curie_data_580ika=curie_data_580ika.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
            curie_data_580izyou=curie_data_580izyou.merge(est_data_plane[['x', 'y', grad_target]], how='left', on=['x', 'y'])
            master_features += [grad_target]
        
        # main
        features = master_features.copy()
        print(features)
        target = 't'

        X_train = train_data[features]
        Y_train = train_data[target]

        X_test = test_data[features]
        Y_test = test_data[target]

        X_est = est_data[features]

        X_curie = curie_data[features]
        Y_curie = curie_data[target]

        X_curie_580ika=curie_data_580ika[features]
        X_curie_580izyou=curie_data_580izyou[features]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_est = scaler.transform(X_est)
        X_curie = scaler.transform(X_curie)
        X_curie_580ika=scaler.transform(X_curie_580ika)
        X_curie_580izyou=scaler.transform(X_curie_580izyou)

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
        X_curie = Variable(torch.from_numpy(X_curie).float()).to(device)
        Y_curie = Variable(torch.from_numpy(
            Y_curie.values.reshape(-1, 1)).float()).to(device)
        X_curie_580ika = Variable(torch.from_numpy(X_curie_580ika).float()).to(device)
        X_curie_580izyou = Variable(torch.from_numpy(X_curie_580izyou).float()).to(device)

        loss_list = []
        learning_curve_list=[]
        model = Net(X_train.shape[1], 1, 120).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        model.train()

        time_start = time.time()
        for epoch in range(2001):
            model.train()
            pred_est = convert_data(X_est)
            pred_train = convert_data(X_train)
            pred_test = convert_data(X_test)
            pred_curie = convert_data(X_curie)
            pred_curie_580ika=convert_data(X_curie_580ika)
            pred_curie_580izyou=convert_data(X_curie_580izyou)

            loss = F.mse_loss(pred_train, Y_train)
            loss_v = F.mse_loss(pred_test, Y_test)
            loss_curie = F.mse_loss(pred_curie, Y_curie)

            if epoch < 0:
                loss_sv=torch.tensor(0)
                loss_sum = loss+Min_t(pred_est, 0)+(1/1000)*loss_curie+Max_t_580(pred_curie_580ika)+(1/1000)*Min_t_580(pred_curie_580izyou)+Max_t(pred_curie_580izyou,800)
            else:
                pc_est = torch.cat((est_data_origin, pred_est), 1)
                loss_sv = []
                for region in ['hokkaidou', 'higashi', 'kyusyu']:
                    sv = calc_pred_semivariogram_xy(
                        pc_est, d_bool_dict, z_unique, region)
                    loss_sv.append(torch.sqrt(
                        F.mse_loss(sv, model_sv_dict[region])))
                loss_sv = torch.mean(torch.stack(loss_sv))
                
                sv_z = calc_pred_semivariogram_z(pc_est,est_data_origin_xy,dist_z_list)
                loss_sv_z=F.mse_loss(sv_z[~model_sv_z.isnan()], model_sv_z[~model_sv_z.isnan()])
                # loss_sv_z=torch.sqrt(loss_sv_z)
                
                if epoch%10 ==0:
                    print(epoch,loss_v.data.item(),loss_sv.data.item(),loss_sv_z.data.item())
                loss_sv = loss_sv + loss_sv_z
                
                loss_sum = loss+Min_t(pred_est, 0)+(1/1000)*loss_curie+Max_t_580(pred_curie_580ika)+(1/1000)*Min_t_580(pred_curie_580izyou)+Max_t(pred_curie_580izyou,800)+(1/1000)*loss_sv

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(
                    f'{epoch},{np.sqrt(loss.data.item())},{np.sqrt(loss_v.data.item())},{np.sqrt(loss_curie.data.item())},{np.sqrt(loss_sv.data.item())},time:{(time.time()-time_start)/60}')
                loss_list.append([epoch, np.sqrt(loss.data.item()), np.sqrt(
                    loss_v.data.item()), np.sqrt(loss_curie.data.item()), np.sqrt(loss_sv.data.item())])
                learning_curve_list.append(np.sqrt(loss_v.data.item()))

        model.eval()
        # pred_est_list.append(convert_data(X_est).detach().cpu().numpy())
        # learning_curve_list_master.append(np.array(learning_curve_list))
        pd.DataFrame(np.array(loss_list), columns=['epoch', 'train_loss', 'test_loss', 'curie_loss','sv_loss']).to_csv(
            f'./output/learning_curve/lc_nk_extra_addz_epoch2000_{num}.csv', index=False)
        print('-'*50)
        est_data['t'] = convert_data(X_est).detach().cpu().numpy()
        est_data = est_data[['x', 'y', 'h_z', 't']]
        est_data.to_csv(f'./output/voxler/est_nk_extra_addz_epoch2000_{num}.csv', index=False)
        torch.save(model.state_dict(), f'./output/model/torch_nk_extra_addz_epoch2000_{num}.pth')
        print((time.time()-time_start)/60)
    
# %%
