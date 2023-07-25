#%%
import pandas as pd
import re
import requests
import time
import json
from tqdm import tqdm
#%%
xy_grid_wgs=pd.read_csv('./input_japan/WGS_albers_h_ja_detail.csv')[['ido','keido']]
#%%
all_tishitsu=[]
time_now=time.time()
for i,(ido,keido) in tqdm(enumerate(xy_grid_wgs.values)):
    url=f'https://gbank.gsj.jp/seamless/v2/api/1.2.1/legend.json?point={ido},{keido}'
    res=requests.get(url)
    tishitsu=res.text
    dict_tishitsu=json.loads(tishitsu)
    df_tishitsu=pd.DataFrame(dict_tishitsu.values(),index=dict_tishitsu.keys()).T
    df_tishitsu['ido'],df_tishitsu['keido']=ido,keido
    df_tishitsu=df_tishitsu[['ido','keido','symbol','formationAge_ja','group_ja','lithology_ja']]
    all_tishitsu.append(df_tishitsu)
    time.sleep(1)
    if i!=0 and i%5000==0:
        time.sleep(60*30)
all_tishitsu=pd.concat(all_tishitsu).reset_index(drop=True)
all_tishitsu.to_csv('./input_japan/grid_tishitsu_ja_detail_idokeido.csv',index=False)
print((time.time()-time_now))
#%%