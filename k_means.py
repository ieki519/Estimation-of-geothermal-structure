#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# %%
def preprocess_input(df):
    add_input_volcano = pd.read_csv('./input/volcano/add_input_volcano.csv')
    add_input_curie = pd.read_csv('./input/curie_point/add_input_curie.csv')
    add_input_tishitsu = pd.read_csv('./input/tishitsu/add_input_tishitsu_pred.csv')
    add_input_onsen = pd.read_csv('./input/onsen/add_input_onsen.csv')

    df = df.merge(add_input_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_input_curie, how='left', on=['x', 'y'])
    df = df.merge(add_input_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_input_onsen, how='left', on=['x', 'y'])

    return df

def preprocess_grid(df):
    add_grid_volcano = pd.read_csv('./input/volcano/add_grid_volcano.csv')
    add_grid_curie = pd.read_csv('./input/curie_point/add_grid_curie.csv')
    add_grid_tishitsu = pd.read_csv('./input/tishitsu/add_grid_tishitsu_pred.csv')
    add_grid_onsen = pd.read_csv('./input/onsen/add_grid_onsen.csv')

    df = df.merge(add_grid_volcano, how='left', on=['x', 'y'])
    df = df.merge(add_grid_curie, how='left', on=['x', 'y'])
    df = df.merge(add_grid_tishitsu, how='left', on=['x', 'y'])
    df = df.merge(add_grid_onsen, how='left', on=['x', 'y'])
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
def feature_engineer(df):
    df['age']=(df['age_a']+df['age_b'])/2
    return df
# %%
input_data = pd.read_csv('./input/useful_data/input_data.csv')
grid_data=pd.read_csv('./input/useful_data/est_grid_500.csv')
grid_data
#%%
input_data=preprocess_input(input_data)
grid_data=preprocess_grid(grid_data)
# input_data = grad_calc(input_data)
# input_data = grad_maxmin_calc(input_data)
grid_data.columns
# %%
#%%
xy_input_data = input_data.groupby(['x', 'y'], as_index=False).mean()
xy_input_data = xy_input_data[['x', 'y', 'h','volcano', 'curie', 'age_a', 'age_b','symbol_freq', 'group_freq', 'lithology_freq', 'Temp', 'pH', 'Na', 'K','Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion']]
xy_input_data=feature_engineer(xy_input_data)
xy_input_data.describe()
#%%
xy_grid_data = grid_data.groupby(['x', 'y'], as_index=False).mean()
xy_grid_data = xy_grid_data[['x', 'y', 'h','volcano', 'curie', 'age_a', 'age_b','symbol_freq', 'group_freq', 'lithology_freq', 'Temp', 'pH', 'Na', 'K','Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'anion']]
xy_grid_data=feature_engineer(xy_grid_data)
xy_grid_data
#%%
features = ['x', 'y', 'h', 'volcano', 'curie', 'Temp', 'pH', 'Na', 'K', 'Ca', 'Mg',
       'Cl', 'SO4', 'HCO3', 'anion', 'age']

xy_input=xy_input_data[features]
xy_grid=xy_grid_data[features]
ss=StandardScaler()

xy_grid=ss.fit_transform(xy_grid)
xy_input=ss.transform(xy_input)

km = KMeans(n_clusters=5)
km.fit(xy_grid)

xy_grid_data['kmeans']=km.predict(xy_grid)
xy_input_data['kmeans']=km.predict(xy_input)

add_grid_kmeans=xy_grid_data[['x','y','kmeans']]
add_input_kmeans=xy_input_data[['x','y','kmeans']]
add_grid_kmeans
fig, ax = plt.subplots()
add_grid_kmeans.plot.scatter('x', 'y', s=2, cmap='brg', colorbar=True,
                    c='kmeans', alpha=1, marker='s', ax=ax)
# %%
add_input_kmeans.to_csv('./input/k_means/add_input_kmeans.csv',index=False)
add_grid_kmeans.to_csv('./input/k_means/add_grid_kmeans.csv',index=False)
add_input_kmeans
# %%
