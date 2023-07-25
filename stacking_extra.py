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

def torch_select_column_by_condition_2(data, index, condition):
    condition_data = data[:, index]
    mask = condition_data.eq(condition)
    mask, mask_ = mask.min(1)
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
    
class StackingNet(nn.Module):
    def __init__(self, input_size, output_size, unit_size):
        super(StackingNet, self).__init__()
        self.fc_s = nn.Linear(input_size, int(unit_size))
        self.fc2 = nn.Linear(int(unit_size), int(unit_size))
        self.fc3 = nn.Linear(int(unit_size), int(unit_size))
        self.fc_e = nn.Linear(int(unit_size), output_size)

    def forward(self, x):

        x = F.relu(self.fc_s(x))
        x=F.dropout(x,p=0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc3(x))
        x=F.dropout(x,p=0.2)
        x = self.fc_e(x)

        return x
#%%
time_start=time.time()
pred_est_list=[]
learning_curve_list_master=[]

fix_seed(0)

for num in range(5):
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
    # model_sv_dict = {}
    # for region in ['hokkaidou', 'higashi', 'kyusyu']:
    #     model_sv = np.array(
    #         calc_model_semivariogram_xy(train_data, region))
    #     model_sv_dict[region] = Variable(
    #         torch.from_numpy(model_sv).float()).to(device)
    # est_data_origin = est_data[['x', 'y', 'h_z']].values
    # est_data_origin = Variable(torch.from_numpy(
    #     est_data_origin).float()).to(device)
    # z_unique = torch.unique(est_data_origin[:, 2])
    # d_bool_dict = create_d_bool_dict(est_data_origin)
    
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
    stacking_features=master_features.copy()#['x', 'y', 'h', 'z', 'h_z']
    print(stacking_features)
    target = 't'

    X_train = train_data[stacking_features]
    Y_train = train_data[target]

    X_test = test_data[stacking_features]
    Y_test = test_data[target]

    X_est = est_data[stacking_features]

    X_curie = curie_data[stacking_features]
    Y_curie = curie_data[target]

    X_curie_580ika=curie_data_580ika[stacking_features]
    X_curie_580izyou=curie_data_580izyou[stacking_features]
    
    X_stacking_raw = pd.concat([train_data,curie_data],axis=0)
    X_stacking= X_stacking_raw[stacking_features]
    Y_stacking= X_stacking_raw[target].values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_est = scaler.transform(X_est)
    X_curie = scaler.transform(X_curie)
    X_curie_580ika=scaler.transform(X_curie_580ika)
    X_curie_580izyou=scaler.transform(X_curie_580izyou)
    
    X_stacking = scaler.transform(X_stacking)

    log_target = np.log(Y_train)
    log_mean = log_target.mean()
    log_std = log_target.std()
    omin = 0
    
    X_test = Variable(torch.from_numpy(X_test).float()).to(device)
    Y_test = Variable(torch.from_numpy(
        Y_test.values.reshape(-1, 1)).float()).to(device)
    X_est = Variable(torch.from_numpy(X_est).float()).to(device)
    X_curie = Variable(torch.from_numpy(X_curie).float()).to(device)
    Y_curie = Variable(torch.from_numpy(
        Y_curie.values.reshape(-1, 1)).float()).to(device)
    X_curie_580ika = Variable(torch.from_numpy(X_curie_580ika).float()).to(device)
    X_curie_580izyou = Variable(torch.from_numpy(X_curie_580izyou).float()).to(device)
    
    # stacking prepare
    stacking_columns = [0,"dnn","rf"]

    X_train = pd.DataFrame(X_train) 
    train_features=pd.DataFrame(np.zeros((X_train.shape[0],len(stacking_columns))),columns=stacking_columns)
    
    test_features=pd.DataFrame(np.zeros((X_test.shape[0],len(stacking_columns))),columns=stacking_columns)
    est_features=pd.DataFrame(np.zeros((X_est.shape[0],len(stacking_columns))),columns=stacking_columns)
    curie_features=pd.DataFrame(np.zeros((X_curie.shape[0],len(stacking_columns))),columns=stacking_columns)
    curie_580ika_features=pd.DataFrame(np.zeros((X_curie_580ika.shape[0],len(stacking_columns))),columns=stacking_columns)
    curie_580izyou_features=pd.DataFrame(np.zeros((X_curie_580izyou.shape[0],len(stacking_columns))),columns=stacking_columns)
    
    input_xy=train_data.groupby(["x","y"],as_index=False).mean()[["x","y"]].reset_index(drop=True)
    idx_list=[]
    for x,y in input_xy.values:
        idx_list.append(train_data[(train_data.x==x) & (train_data.y==y)].index.tolist())
    
    train_features2=pd.DataFrame(np.zeros((X_train.shape[0],1)))
    cv_s = KFold(n_splits=5, shuffle=True, random_state=0)
    for trn_idx, val_idx in cv_s.split(input_xy):
        trn_idx_s=[]
        for idx in trn_idx:
            trn_idx_s+=idx_list[idx]
        val_idx_s=[]
        for idx in val_idx:
            val_idx_s+=idx_list[idx]
            
        # stacking tree train
        trn_x_tree = X_stacking[trn_idx_s,:]
        trn_y_tree = Y_stacking[trn_idx_s]
        val_x_tree = X_stacking[val_idx_s,:]
        rf=RandomForestRegressor(random_state=0)
        rf.fit(trn_x_tree,trn_y_tree)
        train_features.iloc[val_idx_s, stacking_columns.index("rf")]=rf.predict(val_x_tree)
        # xgb=XGBRegressor(random_state=0)
        # xgb.fit(trn_x_tree,trn_y_tree)
        # train_features.iloc[val_idx_s, stacking_columns.index("xgb")]=xgb.predict(val_x_tree)
        # lgbm=LGBMRegressor(random_state=0)
        # lgbm.fit(trn_x_tree,trn_y_tree)
        # train_features.iloc[val_idx_s, stacking_columns.index("lgbm")]=lgbm.predict(val_x_tree)
        
        trn_x_s = X_train.iloc[trn_idx_s,:]
        trn_y_s = Y_train.iloc[trn_idx_s]
        val_x_s = X_train.iloc[val_idx_s,:]
    
        trn_x_s = Variable(torch.from_numpy(trn_x_s.values).float()).to(device)
        trn_y_s = Variable(torch.from_numpy(
            trn_y_s.values.reshape(-1, 1)).float()).to(device)
        val_x_s = Variable(torch.from_numpy(val_x_s.values).float()).to(device)
    
        model = StackingNet(trn_x_s.shape[1], 1, 30).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        for epoch in range(501):
            pred_est = convert_data(X_est)
            pred_train = convert_data(trn_x_s)
            pred_test = convert_data(X_test)
            pred_curie = convert_data(X_curie)
            pred_curie_580ika=convert_data(X_curie_580ika)
            pred_curie_580izyou=convert_data(X_curie_580izyou)

            loss = F.mse_loss(pred_train, trn_y_s)
            loss_v = F.mse_loss(pred_test, Y_test)
            loss_curie = F.mse_loss(pred_curie, Y_curie)
            
            l2 = torch.tensor(0., requires_grad=True)
            for w in model.parameters():
                l2 = l2 + torch.norm(w)**2

            loss_sum = loss+Min_t(pred_est, 0)+(1/1000)*loss_curie+Max_t_580(pred_curie_580ika)+(1/1000)*Min_t_580(pred_curie_580izyou)+Max_t(pred_curie_580izyou,800)+l2

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
        model.eval()
        train_features.iloc[val_idx_s,[stacking_columns.index("dnn")]] = convert_data(val_x_s).detach().cpu().numpy()
    
    MLA = {'rf': RandomForestRegressor(random_state=0)}#,'xgb': XGBRegressor(random_state=0), 'lgbm': LGBMRegressor(random_state=0)}
    for mla in MLA.keys():
        tree=MLA[mla]
        tree.fit(X_stacking,Y_stacking)
        test_features[mla] = tree.predict(X_test.detach().cpu().numpy())
        est_features[mla] = tree.predict(X_est.detach().cpu().numpy())
        curie_features[mla] = tree.predict(X_curie.detach().cpu().numpy())
        curie_580ika_features[mla] = tree.predict(X_curie_580ika.detach().cpu().numpy())
        curie_580izyou_features[mla] = tree.predict(X_curie_580izyou.detach().cpu().numpy())
    
    trn_x_s = Variable(torch.from_numpy(X_train.values).float()).to(device)
    trn_y_s = Variable(torch.from_numpy(Y_train.values.reshape(-1, 1)).float()).to(device)
    model = StackingNet(trn_x_s.shape[1], 1, 30).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for epoch in range(501):
        pred_est = convert_data(X_est)
        pred_train = convert_data(trn_x_s)
        pred_test = convert_data(X_test)
        pred_curie = convert_data(X_curie)
        pred_curie_580ika=convert_data(X_curie_580ika)
        pred_curie_580izyou=convert_data(X_curie_580izyou)

        loss = F.mse_loss(pred_train, trn_y_s)
        loss_v = F.mse_loss(pred_test, Y_test)
        loss_curie = F.mse_loss(pred_curie, Y_curie)
        
        l2 = torch.tensor(0., requires_grad=True)
        for w in model.parameters():
            l2 = l2 + torch.norm(w)**2

        loss_sum = loss+Min_t(pred_est, 0)+(1/1000)*loss_curie+Max_t_580(pred_curie_580ika)+(1/1000)*Min_t_580(pred_curie_580izyou)+Max_t(pred_curie_580izyou,800)+l2

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(
                f'stacking,{epoch},{np.sqrt(loss.data.item())},{np.sqrt(loss_v.data.item())}')
    model.eval()
    test_features["dnn"] = convert_data(X_test).detach().cpu().numpy()
    est_features["dnn"] = convert_data(X_est).detach().cpu().numpy()
    curie_features["dnn"] = convert_data(X_curie).detach().cpu().numpy()
    curie_580ika_features["dnn"] = convert_data(X_curie_580ika).detach().cpu().numpy()
    curie_580izyou_features["dnn"] = convert_data(X_curie_580izyou).detach().cpu().numpy()
    
    train_features.to_csv(f"./input/stacking_features/train_features_{num}.csv",index=False)
    test_features.to_csv(f"./input/stacking_features/test_features_{num}.csv",index=False)
    est_features.to_csv(f"./input/stacking_features/est_features_{num}.csv",index=False)
    curie_features.to_csv(f"./input/stacking_features/curie_features_{num}.csv",index=False)
    curie_580ika_features.to_csv(f"./input/stacking_features/curie_580ika_features_{num}.csv",index=False)
    curie_580izyou_features.to_csv(f"./input/stacking_features/curie_580izyou_features_{num}.csv",index=False)
    
    train_features=pd.read_csv(f"./input/stacking_features/train_features_{num}.csv")
    test_features=pd.read_csv(f"./input/stacking_features/test_features_{num}.csv")
    est_features=pd.read_csv(f"./input/stacking_features/est_features_{num}.csv")
    curie_features=pd.read_csv(f"./input/stacking_features/curie_features_{num}.csv")
    curie_580ika_features=pd.read_csv(f"./input/stacking_features/curie_580ika_features_{num}.csv")
    curie_580izyou_features=pd.read_csv(f"./input/stacking_features/curie_580izyou_features_{num}.csv")
    
    features = master_features.copy()
    X_train = pd.concat([train_data[features],train_features],axis=1)
    print(X_train.columns)
    Y_train = train_data[target]

    X_test = pd.concat([test_data[features],test_features],axis=1)
    Y_test = test_data[target]

    X_est = pd.concat([est_data[features],est_features],axis=1)

    X_curie = pd.concat([curie_data[features],curie_features],axis=1)
    Y_curie = curie_data[target]

    X_curie_580ika=pd.concat([curie_data_580ika[features],curie_580ika_features],axis=1)
    X_curie_580izyou=pd.concat([curie_data_580izyou[features],curie_580izyou_features],axis=1)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_est = scaler.transform(X_est)
    X_curie = scaler.transform(X_curie)
    X_curie_580ika=scaler.transform(X_curie_580ika)
    X_curie_580izyou=scaler.transform(X_curie_580izyou)
    
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
    for epoch in range(2001):
        pred_est = convert_data(X_est)
        pred_train = convert_data(X_train)
        pred_test = convert_data(X_test)
        pred_curie = convert_data(X_curie)
        pred_curie_580ika=convert_data(X_curie_580ika)
        pred_curie_580izyou=convert_data(X_curie_580izyou)

        loss = F.mse_loss(pred_train, Y_train)
        loss_v = F.mse_loss(pred_test, Y_test)
        loss_curie = F.mse_loss(pred_curie, Y_curie)

        loss_sum = loss+Min_t(pred_est, 0)+(1/1000)*loss_curie+Max_t_580(pred_curie_580ika)+(1/1000)*Min_t_580(pred_curie_580izyou)+Max_t(pred_curie_580izyou,800)
        # l2 = torch.tensor(0., requires_grad=True)
        # for w in model.parameters():
        #     l2 = l2 + torch.norm(w)**2
        
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(
                f'{epoch},{np.sqrt(loss.data.item())},{np.sqrt(loss_v.data.item())}')#,{np.sqrt(loss_curie.data.item())}')
            loss_list.append([epoch, np.sqrt(loss.data.item()), np.sqrt(
                loss_v.data.item()), np.sqrt(loss_curie.data.item())])
            learning_curve_list.append(np.sqrt(loss_v.data.item()))
    model.eval()
    pred_est_list.append(convert_data(X_est).detach().cpu().numpy())
    learning_curve_list_master.append(np.array(learning_curve_list))
    pd.DataFrame(np.array(loss_list), columns=['epoch', 'train_loss', 'test_loss', 'curie_loss']).to_csv(
        f'./output/learning_curve/lc_dnn_extra_stacking_{num}.csv', index=False)
    torch.save(model.state_dict(), f'./output/model/torch_dnn_extra_stacking_{num}.pth')
    print('-'*50)


pred_est = np.hstack(pred_est_list).mean(axis=1)
est_data['t'] = pred_est
est_data = est_data[['x', 'y', 'h_z', 't']]
est_data.to_csv(f'./output/voxler/est_dnn_extra_stacking.csv', index=False)
learning_curve=np.vstack(learning_curve_list_master).mean(axis=0)
learning_curve=np.vstack([np.arange(0,2001,100),learning_curve])
pd.DataFrame(learning_curve.T,columns=['epoch', 'test_loss']).to_csv(f'./output/learning_curve/lc_dnn_extra_stacking.csv', index=False)
print((time.time()-time_start)/60)

# %%
