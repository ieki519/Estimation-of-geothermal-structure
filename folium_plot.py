#%%
import folium
import pandas as pd

# このCSVには、県庁所在地の緯度・経度がlatitudeカラムとlongitudeカラムに入っている。
input_data = pd.read_csv('./input_japan/useful_data/database/profile_db_ja.csv')[["x","y","t","ido","keido"]]
# albers_wgs=pd.read_csv("./input/WGS_to_albers/input_WGS_albers.csv")
# albers_wgs.x,albers_wgs.y=albers_wgs.x.round(),albers_wgs.y.round()
input_data=input_data.groupby(["x","y"],as_index=False).max().reset_index(drop=True)
# input_data=input_data.merge(albers_wgs,on=["x","y"],how="left")
input_data=input_data[input_data.t>300].reset_index(drop=True)
input_data
#%%

def visualize_locations(df,  zoom=4):
    """日本を拡大した地図に、pandasデータフレームのlatitudeおよびlongitudeカラムをプロットする。
    """
        	
    # 図の大きさを指定する。
    f = folium.Figure(width=1000, height=500)

    # 初期表示の中心の座標を指定して地図を作成する。
    center_lat=34.686567
    center_lon=135.52000
    m = folium.Map([center_lat,center_lon], zoom_start=zoom).add_to(f)
        
    # データフレームの全ての行のマーカーを作成する。
    for i in range(0,len(df)):
        folium.Marker(location=[df["ido"][i],df["keido"][i]]).add_to(m)
        
    return m
#%%
visualize_locations(input_data)
# %%
input_data["ido"]
# %%
