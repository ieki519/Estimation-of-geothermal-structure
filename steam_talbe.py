#%%
from selenium import webdriver
import numpy as np
import time
# %%
browser = webdriver.Chrome("c:\driver\chromedriver.exe")
# browser = webdriver.ChromeOptions("c:\Program Files (x86)\Google\Chrome\Application\chrome.exe")
URL = "https://www.tlv.com/ja/steam-info/steam-table/"
browser.get(URL)
# %%
for value in [100,200,500]:
    browser.find_element_by_xpath("/html/body/div[2]/div/main/article/div[2]/div/div[1]/div/section[2]/form/div/div[2]/input").clear()
    browser.find_element_by_xpath("/html/body/div[2]/div/main/article/div[2]/div/div[1]/div/section[2]/form/div/div[2]/input").send_keys(value)
    browser.find_element_by_xpath("/html/body/div[2]/div/main/article/div[2]/div/div[1]/div/section[2]/form/ul/li/input").click()
    steam=browser.find_element_by_xpath("/html/body/div[2]/div/main/article/div[2]/div/div[1]/div/section[2]/div").text
    if steam == '不正な値が入力されました。':
        print(steam,value)
        steam = np.nan
    else:
        if steam.split("\n")[2].split(" ")[0]=='飽和水の比エンタルピ':
            steam=float(steam.split("\n")[2].split(" ")[1])
            print(steam,value)
        else:
            print(steam,value)
    time.sleep(0.2)
# %%

# %%
