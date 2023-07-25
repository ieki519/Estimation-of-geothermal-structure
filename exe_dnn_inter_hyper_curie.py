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
    # add_input_volcano = pd.read_csv('./input/volcano/add_input_volcano.csv')
    # add_input_curie = pd.read_csv('./input/curie_point/add_input_curie.csv')
    # add_input_tishitsu = pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
    # add_input_onsen = pd.read_csv('./input/onsen/add_input_onsen.csv')

    # df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    # df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    # df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    # df = df.merge(add_input_onsen, how='left', on=['x', 'y'])

    # # df['group_rank']=df['group_ja'].replace({"火成岩":1,"堆積岩":2,"その他":3,"付加体":4,"変成岩":5})
    # tmp = pd.get_dummies(df["group_ja"], "group_ja")
    # df=pd.concat([df,tmp],axis=1)
    
    # df=df.drop(['symbol','symbol_freq','formationAge_ja', 'group_ja','group_freq', 'lithology_ja', 'lithology_freq'],axis=1)
    # df['age']=(df['age_a']+df['age_b'])/2

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

#%%
# for curie_num in [0,1,10]:
# for curie_num in [100,1000,10000]:
for curie_num in [100000,1000000]:
    time_start = time.time()
    
    fix_seed(0)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    input_data = pd.read_csv('./input_japan/useful_data/input_data_ja.csv')
    input_data = preprocess_input(input_data)

    pred_est_list = []
    learning_curve_list_master=[]

    xy_data = input_data.groupby(
        ['x', 'y'], as_index=False).mean().loc[:, ['x', 'y']]
    for num, (idx_trn, idx_tst) in enumerate(cv.split(xy_data)):
        fix_seed(0)
        # print(seed)
        master_features=['x', 'y', 'h', 'z', 'h_z']
        name = f"basic_curie{curie_num}"
        
        # preprocess
        est_data = pd.read_csv('./input_japan/useful_data/est_grid_500_ja.csv')
        curie_data = pd.read_csv('./input_japan/curie_point/grid_curie_ja.csv')
        curie_data_580ika = pd.read_csv('./input_japan/curie_point/grid_curie_580ika_ja.csv')
        curie_data_580izyou = pd.read_csv('./input_japan/curie_point/grid_curie_580izyou_ja.csv')

        # train test split
        trn_xy = xy_data.iloc[idx_trn, :].values
        tst_xy = xy_data.iloc[idx_tst, :].values
        train_data = pd.DataFrame()
        for x, y in trn_xy:
            trn_data = input_data[(input_data['x'] == x)
                                    & (input_data['y'] == y)]
            train_data = pd.concat([train_data, trn_data], axis=0)
        test_data = pd.DataFrame()
        for x, y in tst_xy:
            tst_data = input_data[(input_data['x'] == x)
                                    & (input_data['y'] == y)]
            test_data = pd.concat([test_data, tst_data], axis=0)
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
        model = Net(X_train.shape[1], 1, 150).to(device)
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

            if curie_num ==0:
                loss_sum = loss+Min_t(pred_est, 0)#+Max_t_580(pred_curie_580ika)#+Max_t(pred_curie_580izyou,800)
            else:
                loss_sum = loss+Min_t(pred_est, 0)+(1/curie_num)*loss_curie+(1/curie_num)*Max_t_580(pred_curie_580ika)+(1/curie_num)*Min_t_580(pred_curie_580izyou)#+Max_t(pred_curie_580izyou,800)
            # 
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(
                    f'{epoch},{np.sqrt(loss.data.item())},{np.sqrt(loss_v.data.item())},{np.sqrt(loss_curie.data.item())}')
                loss_list.append([epoch, np.sqrt(loss.data.item()), np.sqrt(
                    loss_v.data.item()), np.sqrt(loss_curie.data.item())])
                learning_curve_list.append(np.sqrt(loss_v.data.item()))
        model.eval()
        pred_est_list.append(convert_data(X_est).detach().cpu().numpy())
        learning_curve_list_master.append(np.array(learning_curve_list))
        pd.DataFrame(np.array(loss_list), columns=['epoch', 'train_loss', 'test_loss', 'curie_loss']).to_csv(
            f'./output_japan_last/learning_curve/curie_hyper/lc_dnn_inter_{name}_{num}.csv', index=False)
        torch.save(model.state_dict(), f'./output_japan_last/model/torch_dnn_inter_{name}_{num}.pth')
        print('-'*50)
    pred_est = np.hstack(pred_est_list).mean(axis=1)
    est_data['t'] = pred_est
    est_data = est_data[['x', 'y', 'h_z', 't']]
    est_data.to_csv(f'./output_japan_last/voxler/est_dnn_inter_{name}.csv', index=False)
    learning_curve=np.vstack(learning_curve_list_master).mean(axis=0)
    learning_curve=np.vstack([np.arange(0,2001,100),learning_curve])
    pd.DataFrame(learning_curve.T,columns=['epoch', 'test_loss']).to_csv(f'./output_japan_last/learning_curve/curie_hyper/lc_dnn_inter_{name}.csv', index=False)
    print((time.time()-time_start)/60)
# %%
