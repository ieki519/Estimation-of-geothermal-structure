#%%
import pandas as pd
# %%
input_data = pd.read_csv("./input_japan/useful_data/input_data_ja.csv")
input_data.describe()
# %%
import codecs
with codecs.open(rf"./database_ja/火山/火山名.csv",'r','shift-jis','ignore') as f:
    df=pd.read_csv(f)
df
# %%
pd.DataFrame(df.values.reshape(-1,5)).to_csv("./database_ja/火山/火山名5.csv",index=False,encoding="cp932")
# %%
