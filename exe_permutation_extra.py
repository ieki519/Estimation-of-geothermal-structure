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

def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('./input_japan/volcano/add_grid_volcano_detail_ja.csv')
    add_grid_curie = pd.read_csv('./input_japan/curie_point/add_grid_curie_detail_ja.csv')
    add_grid_tishitsu = pd.read_csv('./input_japan/tishitsu/add_grid_tishitsu_detail_pred_ja.csv')
    add_grid_onsen = pd.read_csv('./input_japan/onsen/add_grid_onsen_detail_ja.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])

    # df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    tmp = pd.get_dummies(df["group_ja"], "group_ja")
    df=pd.concat([df,tmp],axis=1)
    
    df=df.drop(['symbol','symbol_freq','formationAge_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)#, 'group_ja'
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

def permutation_importance(df,except_list):
    df = df.copy()
    df_plane = df.groupby(['x', 'y'], as_index=False).mean()
    df_plane.loc[:,except_list] = df_plane[except_list].sample(frac=1,random_state=0).reset_index(drop=True)
    df.drop(except_list,axis=1,inplace=True)
    df = df.merge(df_plane[["x","y"]+except_list],how="left",on=["x","y"])
    return df

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
# master_features_dict={"only_base":['x', 'y', 'h', 'z', 'h_z'],
#                     "only_volcano":['x', 'y', 'h', 'z', 'h_z',"volcano"],
#                     "only_curie":['x', 'y', 'h', 'z', 'h_z',"curie"],
#                     "only_tishitsu":['x', 'y', 'h', 'z', 'h_z', 'age_a', 'age_b', 'group_rank', 'age'],
#                     "only_onsen":['x', 'y', 'h', 'z', 'h_z','Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
#                     "only_grad":['x', 'y', 'h', 'z', 'h_z'],
#                     'only_depth':['x', 'y', 'h', 'z', 'h_z'],
#                     "all":['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
#                             'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
#                              'group_rank', 'age']
#                         }
#%%
# master_features_dict={
#                     "except_volcano":['x', 'y', 'h', 'z', 'h_z', 'curie', 'age_a', 'age_b',
#                             'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
#                              'group_rank', 'age'],
#                     "except_curie":['x', 'y', 'h', 'z', 'h_z', 'volcano', 'age_a', 'age_b',
#                             'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
#                              'group_rank', 'age'],
#                     "except_tishitsu":['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie',
#                             'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
#                     "except_onsen":['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
#                              'group_rank', 'age'],
#                     "except_grad":['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
#                             'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
#                              'group_rank', 'age'],
#                     'except_depth':['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
#                             'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
#                              'group_rank', 'age']
#                         }
#%%
# master_features_dict={
#                     "except_volcano2":['x', 'y', 'h', 'z', 'h_z', 'curie', 'age_a', 'age_b',
#                             'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
#                              'group_rank', 'age'],
#                     "except_curie2":['x', 'y', 'h', 'z', 'h_z', 'volcano', 'age_a', 'age_b',
#                             'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
#                              'group_rank', 'age'],
#                     "except_tishitsu2":['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie',
#                             'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
#                     "except_onsen2":['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
#                              'group_rank', 'age'],
#                     'except_depth2':['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
#                             'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
#                              'group_rank', 'age']
#                     }

#%%
# master_features_dict={
#                     "all":['x', 'y', 'h', 'z', 'h_z', 'volcano', 'curie', 'age_a', 'age_b',
#                             'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion',
#                              'group_rank', 'age']
#                         }
#%%
# master_except_features_dict={
#                     "except_volcano":['volcano'],
#                     "except_curie":['curie'],
#                     "except_onsen":['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
#                     "except_tishitsu":['age_a', 'age_b','age','group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩'],
#                     "except_depth":["depth0","depth500","depth1000"],#extraは800
#                     "except_grad":['grad','grad_max','grad_min'],
#                     "all":[]
#                     }
#%%
features_dict = {
                "volcano":['volcano'],
                "curie":['curie'],
                "onsen":['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],
                "tishitsu_ohe":['age_a', 'age_b','age'],
                "tishitsu_rank":['age_a', 'age_b','age',"group_rank"],
                "depth800":[],#extraは800
                "grad":[],
                }
#%%
# import argparse
# parser = argparse.ArgumentParser() 
# parser.add_argument("--number", type=int)
# args = parser.parse_args() 
# print(args.number)
#%%
permutation_dict={}
target_feature="grad"
#%%
# for except_f in ["volcano","curie",['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],['age_a', 'age_b','age',"group_rank"],["depth0","depth500","depth800"],['grad','grad_max','grad_min','grad_max_h_z','grad_min_h_z']]:
for except_f in ["volcano","curie",['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion'],['age_a', 'age_b','age''group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩'],["depth0","depth500","depth1000"],['grad','grad_max','grad_min','grad_max_h_z','grad_min_h_z']]:
# for except_f in ['Temp', 'pH', 'Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion']: 
# for except_f in [['age_a', 'age_b','age'],'group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩']:
# for except_f in [['age_a', 'age_b','age'],"group_rank"]:    
# for except_f in ["depth0","depth500","depth800"]:
# for except_f in ['grad','grad_max','grad_min','grad_max_h_z','grad_min_h_z']:
    time_start = time.time()
    fix_seed(0)

    pred_est_list = []
    learning_curve_list_master=[]

    for num in range(1):
        fix_seed(num)
        
        master_features=['x', 'y', 'h', 'z', 'h_z']
        name = "basic"
        for f in ['volcano', 'curie', 'onsen', 'tishitsu_ohe', 'depth800', 'grad']:#[target_feature]:
            master_features+=features_dict[f]
            name += "_"+f 
            
        if isinstance(except_f,list):
            except_features=except_f
            except_name = except_f[-1]
        else:
            except_features=[except_f]
            except_name = except_f

        print(name)
        cat_features = ['group_ja_その他', 'group_ja_付加体', 'group_ja_堆積岩','group_ja_変成岩', 'group_ja_火成岩']
        # preprocess
        input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
        input_data = preprocess_input(input_data)

        train_data, test_data = extra_split(input_data)
        
        est_data = pd.read_csv('./input_japan/useful_data/est_grid_500_ja.csv')
        curie_data = pd.read_csv('./input_japan/curie_point/grid_curie_ja.csv')
        curie_data_580ika = pd.read_csv('./input_japan/curie_point/grid_curie_580ika_ja.csv')
        curie_data_580izyou = pd.read_csv('./input_japan/curie_point/grid_curie_580izyou_ja.csv')

        est_data = preprocess_grid(est_data)
        curie_data = preprocess_grid(curie_data)
        curie_data_580ika = preprocess_grid(curie_data_580ika)
        curie_data_580izyou = preprocess_grid(curie_data_580izyou)

        
        #tishitsu rank feature
        if "tishitsu_rank" in name:
            tishitsu_rank_dict = train_data.groupby("group_ja").mean()[["t"]].rank(ascending=False).to_dict()["t"]
            train_data["group_rank"]=train_data["group_ja"].replace(tishitsu_rank_dict)
            test_data["group_rank"]=test_data["group_ja"].replace(tishitsu_rank_dict)
            est_data["group_rank"]=est_data["group_ja"].replace(tishitsu_rank_dict)
            curie_data["group_rank"]=curie_data["group_ja"].replace(tishitsu_rank_dict)
            curie_data_580ika["group_rank"]=curie_data_580ika["group_ja"].replace(tishitsu_rank_dict)
            curie_data_580izyou["group_rank"]=curie_data_580izyou["group_ja"].replace(tishitsu_rank_dict)
        
        #depth feature
        # if "except_depth" not in name:
        # if ("only_depth" in name) or ("all" in name):
        if "depth" in name:
            depth_target_list=[0,500,800]
            depth_features = master_features.copy()
            if "tishitsu_ohe" in name:
                depth_features+=cat_features
            
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
        # if "except_grad" not in name:
        # if ("only_grad" in name) or ("all" in name):
        if "grad" in name:
            train_data = grad_calc(train_data)
            train_data=grad_maxmin_calc(train_data)

            train_data_plane = train_data.groupby(
                ['x', 'y'], as_index=False).mean()
            test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
            est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

            grad_features = master_features.copy()
            
            if "tishitsu_ohe" in name:
                grad_features+=cat_features
                
            grad_target_list = ['grad','grad_max','grad_min','grad_max_h_z','grad_min_h_z']

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
                
        #permutation
        train_data = permutation_importance(train_data,except_features)
        test_data = permutation_importance(test_data,except_features)
        est_data = permutation_importance(est_data,except_features)
        curie_data = permutation_importance(curie_data,except_features)
        curie_data_580ika = permutation_importance(curie_data_580ika,except_features)
        curie_data_580izyou = permutation_importance(curie_data_580izyou,except_features)
        
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
        
        if "tishitsu_ohe" in name:
            print(cat_features)
            X_train = np.hstack((X_train,train_data[cat_features].values))
            X_test = np.hstack((X_test,test_data[cat_features].values))
            X_est = np.hstack((X_est,est_data[cat_features].values))
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
        X_curie = Variable(torch.from_numpy(X_curie).float()).to(device)
        Y_curie = Variable(torch.from_numpy(
            Y_curie.values.reshape(-1, 1)).float()).to(device)
        X_curie_580ika = Variable(torch.from_numpy(X_curie_580ika).float()).to(device)
        X_curie_580izyou = Variable(torch.from_numpy(X_curie_580izyou).float()).to(device)

        loss_list = []
        learning_curve_list=[]
        model = Net(X_train.shape[1], 1, 150).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        model.load_state_dict(torch.load(f'./output_japan_last/model/torch_dnn_extra_{name}_{num}.pth'))
        model.eval()
        pred_test = convert_data(X_test)
        loss_v = F.mse_loss(pred_test, Y_test)
        permutation_dict[f"{name}_except_{except_name}"]=permutation_dict.get(f"{name}_except_{except_name}",[])+[np.sqrt(loss_v.data.item())]
for key in permutation_dict.keys():
    permutation_dict[key]=np.mean(permutation_dict[key])
ep=pd.DataFrame(permutation_dict.items(),columns=["name","test_loss"])
e0=pd.read_csv(f'./output_japan_last/learning_curve/select_features/lc_dnn_extra_{name}_0.csv')["test_loss"].values[-1]
ep["PI"]=ep["test_loss"]/e0
ep.sort_values("PI",ascending=False).to_csv(f"./output_japan_last/PI/{name}_PI_extra.csv",index=False)
# %%


# %%
