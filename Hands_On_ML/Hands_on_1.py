#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
import sklearn
import os
#datapath를 설정해놓음. 나중에 데이터 다운받을 떄 폴더 dataset 안에 lifesat 안에 들어가게 됨.
datapath = os.path.join("datasets", "lifesat", "")


# In[2]:


#pandas의 pivot함수를 사용하여 Indicator 열의 값들을 변수로 올린다. 
#GDP per capita 를 사용하여 Life satisfaction을 나타낼 수 있게 데이터 가공
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# In[3]:


# 주피터에 그래프를 깔끔하게 그리기 위해서
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[6]:


# 데이터 다운로드
import urllib.request
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
#os 패키지를 사용해서 폴더를 만들어준다. 
os.makedirs(datapath, exist_ok=True)
for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
    urllib.request.urlretrieve(url, datapath + filename)


# In[7]:


# 예제 코드
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

# 데이터 적재
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")

# 데이터 준비
# 여기서 np.c_ 는 두 개의 1차원 배열을 세로로 붙혀서 2차원 배열로 만드는 것이다. 
# 이 코드에서는 한 줄의 Series를 2차원 배열로 만들라고 쓰는 것 같음.
# country_stats["GDP per capita"] 랑 X를 출력해서 비교해보면 알기 쉬움
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# 데이터 시각화
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# 선형 모델 선택
model = sklearn.linear_model.LinearRegression()

# 모델 훈련
model.fit(X, y)

# 키프로스에 대한 예측
X_new = [[22587]]  # 키프로스 1인당 GDP
print(model.predict(X_new)) # 출력 [[ 5.96242338]]

