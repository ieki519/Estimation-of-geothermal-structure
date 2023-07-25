# %%
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
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# %%


def preprocess_input(df):
    df['t'] = np.where(df['t'].values <= 0, 0.1, df['t'].values)
    add_input_volcano = pd.read_csv('../input/volcano/add_input_volcano.csv')
    add_input_curie = pd.read_csv('../input/curie_point/add_input_curie.csv')
    add_input_onsen = pd.read_csv('../input/onsen/add_input_onsen_new.csv')

    df['x'], df['y'] = df['x'].round(), df['y'].round()
    add_input_volcano['x'], add_input_volcano['y'] = add_input_volcano['x'].round(
    ), add_input_volcano['y'].round()
    add_input_curie['x'], add_input_curie['y'] = add_input_curie['x'].round(
    ), add_input_curie['y'].round()
    add_input_onsen['x'], add_input_onsen['y'] = add_input_onsen['x'].round(
    ), add_input_onsen['y'].round()

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])

    return df


def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('../input/volcano/add_grid_volcano.csv')
    add_grid_curie = pd.read_csv('../input/curie_point/add_grid_curie.csv')
    add_grid_onsen = pd.read_csv('../input/onsen/add_grid_onsen_new.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])

    return df


def grad_calc(df):
    xy_unique = df.groupby(['x', 'y'], as_index=False).mean()[
        ['x', 'y']].values
    for x, y in xy_unique:
        zt = df[(df['x'] == x) & (df['y'] == y)][['z', 't']]
        grad = np.polyfit(zt['z'], zt['t'], 1)[0]*1000
        df.loc[(df['x'] == x) & (df['y'] == y), 'grad'] = grad
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


class Optimizer:
    def __init__(self, target='temp', trials=500):  # trial_point500
        self.target = target
        self.trials = trials
        self.sampler = TPESampler(seed=0)

    def objective(self, trial):
        model = create_model(trial)
        if self.target == 'grad':
            dict_cv = model_selection.cross_validate(
                model, X_train, Y_train, cv=cv, scoring='neg_root_mean_squared_error')
            return -dict_cv['test_score'].mean()
        else:
            model.fit(X_train, Y_train)
            pred = model.predict(X_test)
            return np.sqrt(mean_squared_error(Y_test, pred))

    def optimize(self):
        study = optuna.create_study(direction="minimize", sampler=self.sampler)
        study.optimize(self.objective, n_trials=self.trials)
        return study.best_params, study.best_value


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


# %%
inter = 0
extra = 1
no = 18
# %%
if inter:
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    input_data = pd.read_csv('../input/useful_data/input_data.csv')
    input_data = preprocess_input(input_data)
    pred_est_list = []

    xy_data = input_data.groupby(
        ['x', 'y'], as_index=False).mean().loc[:, ['x', 'y']]
    for num, (idx_trn, idx_tst) in enumerate(cv.split(xy_data)):
        est_data = pd.read_csv('../input/useful_data/est_grid_500.csv')
        curie_data = pd.read_csv('../input/curie_point/grid_curie.csv')
        est_data = preprocess_grid(est_data)
        curie_data = preprocess_grid(curie_data)

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

        train_data = grad_calc(train_data)
        train_data_plane = train_data.groupby(
            ['x', 'y'], as_index=False).mean()
        test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
        est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

        grad_features = ['x', 'y', 'h', 'volcano', 'curie',
                         'pH',  'Cl', 'SO4', 'HCO3','anion','Temp', 'Na', 'K', 'Ca', 'Mg']
        grad_target = 'grad'
        X_train = train_data_plane[grad_features]
        Y_train = train_data_plane[grad_target]

        X_test = test_data_plane[grad_features]

        X_est = est_data_plane[grad_features]

        X_curie = curie_data[grad_features]

        lgb_params = {'max_depth': 11,
                      'n_estimators': 79,
                      'learning_rate': 0.04614912281574665,
                      'num_leaves': 2139,
                      'min_child_samples': 15,
                      'random_state': 0}
        lgb = LGBMRegressor(**lgb_params)
        lgb.fit(X_train, Y_train)

        test_data_plane['grad'] = lgb.predict(X_test)
        est_data_plane['grad'] = lgb.predict(X_est)
        curie_data['grad'] = lgb.predict(X_curie)

        test_data = test_data.merge(
            test_data_plane[['x', 'y', 'grad']], how='left', on=['x', 'y'])

        est_data = est_data.merge(
            est_data_plane[['x', 'y', 'grad']], how='left', on=['x', 'y'])

        features = ['x', 'y', 'h', 'z', 'h_z']#, 'volcano', 'curie', 'grad',
                     #'pH',  'Cl', 'SO4', 'HCO3','anion','Temp','Na', 'K', 'Ca', 'Mg']
        target = 't'

        X_train = train_data[features]
        Y_train = train_data[target]

        X_test = test_data[features]
        Y_test = test_data[target]

        X_est = est_data[features]

        X_curie = curie_data[features]
        Y_curie = curie_data[target]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_est = scaler.transform(X_est)
        X_curie = scaler.transform(X_curie)

        log_target = np.log(Y_train)
        log_mean = log_target.mean()
        log_std = log_target.std()
        omax = 580
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

        loss_list = []
        model = Net(X_train.shape[1], 1, 120).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        for epoch in range(2001):
            pred_est = convert_data(X_est)
            pred_train = convert_data(X_train)
            pred_test = convert_data(X_test)
            pred_curie = convert_data(X_curie)

            loss = F.mse_loss(pred_train, Y_train)
            loss_v = F.mse_loss(pred_test, Y_test)
            loss_curie = F.mse_loss(pred_curie, Y_curie)

            loss_sum = loss+Max_t(pred_est, omax) + \
                Min_t(pred_est, omin)#+(1/1000)*loss_curie

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(
                    f'{epoch},{np.sqrt(loss.data.item())},{np.sqrt(loss_v.data.item())},{np.sqrt(loss_curie.data.item())}')
                loss_list.append([epoch, np.sqrt(loss.data.item()), np.sqrt(
                    loss_v.data.item()), np.sqrt(loss_curie.data.item())])
        model.eval()
        pred_est_list.append(convert_data(X_est).detach().cpu().numpy())
        pd.DataFrame(np.array(loss_list), columns=['epoch', 'train_loss', 'test_loss', 'curie_loss']).to_csv(
            f'./output/learning_curve/lc_inter_{no}_{num}.csv', index=False)
        print('-'*50)
    pred_est = np.hstack(pred_est_list).mean(axis=1)
    est_data['t'] = pred_est
    est_data = est_data[['x', 'y', 'h_z', 't']]
    est_data.to_csv(f'./output/voxler/est_torch_inter_{no}.csv', index=False)

# %%
if extra:
    pred_est_list = []
    for num in range(1):
        input_data = pd.read_csv('../input/useful_data/input_data.csv')
        input_data = preprocess_input(input_data)

        train_data, test_data = extra_split(input_data)

        est_data = pd.read_csv('../input/useful_data/est_grid_500.csv')
        curie_data = pd.read_csv('../input/curie_point/grid_curie.csv')
        est_data = preprocess_grid(est_data)
        curie_data = preprocess_grid(curie_data)

        train_data = grad_calc(train_data)
        train_data_plane = train_data.groupby(
            ['x', 'y'], as_index=False).mean()
        test_data_plane = test_data.groupby(['x', 'y'], as_index=False).mean()
        est_data_plane = est_data.groupby(['x', 'y'], as_index=False).mean()

        # 
        grad_features = ['x', 'y', 'h', 'volcano', 'curie', 'pH', 'Cl', 'SO4', 'HCO3']#,'Temp', 'Na','K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3']#,'anion']
        grad_target = 'grad'
        X_train = train_data_plane[grad_features]
        Y_train = train_data_plane[grad_target]

        X_test = test_data_plane[grad_features]

        X_est = est_data_plane[grad_features]

        X_curie = curie_data[grad_features]

        lgb_params = {'max_depth': 11,
                      'n_estimators': 79,
                      'learning_rate': 0.04614912281574665,
                      'num_leaves': 2139,
                      'min_child_samples': 15,
                      'random_state': 0}
        lgb = LGBMRegressor(**lgb_params)
        lgb.fit(X_train, Y_train)

        test_data_plane['grad'] = lgb.predict(X_test)
        est_data_plane['grad'] = lgb.predict(X_est)
        curie_data['grad'] = lgb.predict(X_curie)

        test_data = test_data.merge(
            test_data_plane[['x', 'y', 'grad']], how='left', on=['x', 'y'])

        est_data = est_data.merge(
            est_data_plane[['x', 'y', 'grad']], how='left', on=['x', 'y'])

        # 
        features = ['x', 'y', 'h', 'z', 'h_z']#, 'volcano', 'curie', 'grad', 'pH', 'Cl', 'SO4', 'HCO3']#,'Temp', 'Na','K', 'Ca', 'Mg']#,'anion']
        target = 't'

        X_train = train_data[features]
        Y_train = train_data[target]

        X_test = test_data[features]
        Y_test = test_data[target]

        X_est = est_data[features]

        X_curie = curie_data[features]
        Y_curie = curie_data[target]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_est = scaler.transform(X_est)
        X_curie = scaler.transform(X_curie)

        log_target = np.log(Y_train)
        log_mean = log_target.mean()
        log_std = log_target.std()
        omax = 600
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

        loss_list = []
        model = Net(X_train.shape[1], 1, 120).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        for epoch in range(2001):
            pred_est = convert_data(X_est)
            pred_train = convert_data(X_train)
            pred_test = convert_data(X_test)
            pred_curie = convert_data(X_curie)

            loss = F.mse_loss(pred_train, Y_train)
            loss_v = F.mse_loss(pred_test, Y_test)
            loss_curie = F.mse_loss(pred_curie, Y_curie)

            loss_sum = loss+Max_t(pred_est, omax) + \
                Min_t(pred_est, omin)+(1/1000)*loss_curie

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(
                    f'{epoch},{np.sqrt(loss.data.item())},{np.sqrt(loss_v.data.item())},{np.sqrt(loss_curie.data.item())}')
                loss_list.append([epoch, np.sqrt(loss.data.item()), np.sqrt(
                    loss_v.data.item()), np.sqrt(loss_curie.data.item())])
        model.eval()
        pred_est_list.append(convert_data(X_est).detach().cpu().numpy())
        pd.DataFrame(np.array(loss_list), columns=['epoch', 'train_loss', 'test_loss', 'curie_loss']).to_csv(
            f'./output/learning_curve/lc_extra_{no}_{num}.csv', index=False)
        print('-'*50)
    pred_est = np.hstack(pred_est_list).mean(axis=1)
    est_data['t'] = pred_est
    est_data = est_data[['x', 'y', 'h_z', 't']]
    est_data.to_csv(f'./output/voxler/est_torch_extra_{no}.csv', index=False)
# %%
# train_loss_list=[]
# test_loss_list=[]
# for i in range(10):
#     lc=pd.read_csv(f'./output/learning_curve/lc_{i}.csv')
#     epoch=lc['epoch']
#     train_loss=lc['train_loss'].values
#     test_loss=lc['test_loss'].values
#     train_loss_list.append(train_loss)
#     test_loss_list.append(test_loss)
# train_loss_master=np.vstack(train_loss_list).mean(axis=0)
# test_loss_master=np.vstack(test_loss_list).mean(axis=0)

# fig, ax = plt.subplots()
# ax.grid()
# ax.set_xlabel('Num. of Learning Epochs')
# ax.set_ylabel('RMSE(℃)')
# # ax.plot(epoch,test_loss_master_c,color='black')
# ax.plot(epoch,test_loss_master,color='red')
# plt.rcParams["font.size"] = 20
# ax.legend([u'Model_0',u'Model_1'],prop={'family':"MS Gothic"})
# %%
