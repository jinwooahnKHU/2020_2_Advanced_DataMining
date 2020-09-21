#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
import urllib


# In[2]:


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# In[3]:


def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok = True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    #압축풀고 housing.csv파일을 만든다.
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()


# In[4]:


fetch_housing_data()


# In[5]:


import pandas as pd
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[6]:


housing = load_housing_data()


# In[7]:


housing.head()


# In[8]:


housing.info()
#null 값을 알아볼 때 유용
#total_bedrooms를 보면 null인 값들이 몇 개 있음을 알 수 있다.


# In[9]:


#Ocean_proximity 가 객체형태라서 어떻게 되어있는지 확인
housing['ocean_proximity']


# In[10]:


housing['ocean_proximity'].value_counts()


# In[11]:


#describe()는 숫자형 특성의 요약 정보를 보여준다.
housing.describe()


# *** Histogram ***

# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


housing.hist(bins = 50, figsize = (10,5))
plt.show()


# *** Data Snooping 편향을 방지하기 위해 Test Set을 미리 떼어 놓는다. ***

# In[14]:


import numpy as np


# In[15]:


#random.permutation : 배열을 뒤섞어서 객체로 반환
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[: test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[16]:


train_set, test_set = split_train_test(housing, 0.2)


# In[17]:


print(len(train_set), len(test_set))


# *** 업데이트 등의 변형에도 변하지 않는 샘플을 유지하는 법 ***
# => 샘플마다 식별자를 사용한다!

# In[18]:


from zlib import crc32


# In[19]:


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[20]:


housing_with_id = housing.reset_index() # => 원래 데이터에 "index"열이 추가된 df가 반환됨
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# *** Stratified Sampling ***
# 가령, 중간 소득수준이 중간 주택 가격의 예측에 중요한 영향을 미친다고 하자. 이러면, 테스트 세트와 훈련세트의 소득 수준이 비슷한 비율로 존재해야 할 것. 

# In[21]:


#income_cat이라는 파생변수를 만듬. 중간소득수준을 나누어서 라벨링한 값.
housing["income_cat"] = pd.cut(housing['median_income'],
                              bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels = [1,2,3,4,5])


# In[22]:


housing['income_cat'].hist()


# In[23]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[24]:


split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[25]:


strat_test_set['income_cat'].value_counts() / len(strat_test_set)


# In[26]:


#income_cat 특성 삭제
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)


# *** EDA ***

# In[27]:


housing = strat_train_set.copy()


# In[28]:


#지리 정보가 있으니 모든 구역을 산점도로 만들어 시각화해보자
#알파값을 두면 밀집한 부분을 알 수 있다.
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)


# In[29]:


# s : 원의 반지름은 구역의 인구를 나타낸다
# c : 색상을 말하며 가격을 나타낸다.
#미리 정의된 color map 
housing.plot(kind = "scatter", x = "longitude", y = 'latitude', alpha = 0.4,
            s = housing['population']/100, label = "population", figsize = (10,7),
            c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True,
            sharex = False
            )
plt.legend()
plt.show()


# *** 상관관계 조사 ***

# In[30]:


corr_matrix = housing.corr()


# In[32]:


corr_matrix['median_house_value'].sort_values(ascending = False)


# In[34]:


#특성 사이의 상관관계를 확인하는 다른 방법은 숫자형 특성 사이에 산점도를 그려주는 것이다.
from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(housing[attributes], figsize = (10,5))


# In[35]:


housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)


# *** 여러 특성의 조합을 시도해보자 ***

# In[36]:


#특정 구역의 방 개수보다 가구당 방 개수가 더 중요, 가구당 인원도 흥미로운 특성 조합같으니 만들어보자
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']


# In[37]:


#만든 특성 조합이 효과적인지 상관관계 분석을 통해 확인
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)


# *** 알고리즘을 위한 데이터 준비 ***

# In[38]:


#drop()은 데이터 복사본을 만들며 원 데이터에는 영향을 미치지 않는다
#label값을 분리
housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()


# In[39]:


#결측치 대체하기
#앞서 total_bedrooms에 결측치가 있는 것을 확인했었음
#중간값으로 대체해보자
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "median")


# In[41]:


#중간값이 수치형 특성에서만 계산될 수 있기 때문에 text 특성인 ocean_proximity를 제외한 데이터를
#생성하고 거기에 적용하자. 
#비록 결측치는 total_bedrooms에만 있지만 새로운 데이터에서도 누락될 수도 있기에 모든 수치형 특성에
#동일한 imputer을 적용하는 것이 바람직하다
housing_num = housing.drop("ocean_proximity", axis = 1)

imputer.fit(housing_num)


# In[42]:


#각 특성의 중간값을 계산해서 결과를 statistics_ 객체에 저장해놓는다. 
imputer.statistics_


# In[43]:


#학습된 imputer 객체를 사용하여 훈련 세트에서 누락된 값을 학습한 중간값으로 바꿈
X = imputer.transform(housing_num)


# In[44]:


#이 결과는 numpy  배열이므로 dataframe형태로 바꿔주자
type(X)


# In[45]:


housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)


# In[46]:


housing_tr


# *** 텍스트와 범주형 특성 다루기 ***

# In[51]:


# df[[colname]] 이렇게 하면 series가 아니라 dataframe 형식으로 바로 튀어나옴!!
housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)


# In[52]:


#임의의 텍스트가 아니라 범주형 특성임 => 텍스트에서 숫자로 바꾸자
#By using OrdinalEncoder class

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()


# In[54]:


housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[57]:


#categories_ 인스턴스 변수를 통해 카테고리 목록 가져오기
ordinal_encoder.categories_


# 중요!! 그러나 Ordinal Encoder의 경우에는 가까이 있는 두 값이 떨어져 있는 두 값보다 더 비슷하다고 
# 생각한다. 따라서 Ordinal 특성이 아닌 이 경우에는 ONe-Hot Encoding을 해주는 것이 더 바람직하다
# 

# In[58]:


#One hot encoding
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()


# In[59]:


housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[60]:


housing_cat_1hot.toarray()


# In[61]:


cat_encoder.categories_


# *** 나만의 변환기 만들기 ***

# In[66]:


from sklearn.base import BaseEstimator, TransformerMixin


# In[68]:


rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, rooms_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)


# *** 특성 스케일링 ***

# #MinMax scaling vs 표준화
# #Min Max Scaling은 0~1 범위에 들도록 값을 이동하고 스케일을 조정한다. 데이터에서 최솟값을 빼고
# #최댓값과 최솟값의 차이로 나누면 됨.
# #표준화는 평균을 빼고 표준편차로 나누어 분포의 분산이 1이 되도록 한다.
# #표준화는 상한과 하한이 없어 문제가 되는경우도 존재, 그러나 표준화는 이상치에 영향을 덜 받는다.

# *** 복잡한 전처리 과정을 일련의 과정으로 할 수 있게 해주는 파이프 라인 ***

# In[62]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[69]:


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])


# In[70]:


housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[71]:


#열마다 변환을 해주는 작업
from sklearn.compose import ColumnTransformer


# In[72]:


num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)


# *** 모델 선택과 훈련하기 ***

# In[73]:


#회귀분석 모델
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[74]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("예측 : " , lin_reg.predict(some_data_prepared))


# In[75]:


print("레이블: ", list(some_labels))


# In[76]:


#RMSE 측정
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
#예측오차가 68916달러라는 소리.


# In[77]:


#의사결정나무 훈련
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[78]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse  = mean_squared_error(housing_labels, housing_predictions)
tree_mse = np.sqrt(tree_mse)
tree_mse
#과대적합한듯


# 아직까지는 test set을 이용하지 않았음. 확신이 드는 모델의 launch 준비가 되기 전까지는 테스트 세트를
# 사용하지 않고, 훈련세트의 일부분으로 훈련을 하고 다른 일부분은 모델 검증에 사용해야 한다!!
# 

# *** 교차검증을 사용한 평가 ***

# In[79]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring = 'neg_mean_squared_error', cv = 10)
tree_rmse_scores = np.sqrt(-scores)


# In[80]:


def display_scores(scores):
    print("점수: ", scores)
    print("평균: ", scores.mean())
    print("표준편차: ", scores.std())
    
display_scores(tree_rmse_scores)


# *** 모델 세부 튜닝 ***

# In[83]:


from sklearn.ensemble import RandomForestRegressor


# In[84]:


#Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators' : [3, 10, 30], 'max_features' : [2, 4, 6, 8]},
    {'bootstrap' : [False], 'n_estimators' : [3,10], 'max_features' : [2,3,4]},
    
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = 'neg_mean_squared_error',
                          return_train_score = True)

grid_search.fit(housing_prepared, housing_labels)


# In[85]:


grid_search.best_params_


# In[86]:


grid_search.best_estimator_


# In[88]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# *** 최상의 모델과 오차 분석, 특성 중요도 ***

# In[89]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[92]:


extra_attribs = ['rooms_per_hold', "pop_per_hold", 'bedrooms_per_room']
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse = True)


# *** Test Set으로 평가하기!! 이 때 주의할 점은 테스트 세트에서 훈련을 시키면 안되기 때문에 
# fit_transform()이 아니라 transform()을 호출해야함!!!!!! ***

# In[94]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop('median_house_value', axis = 1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)


# In[96]:


#오차가 미세하게 나올경우 신뢰구간까지 측정
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) **2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc = squared_errors.mean(),
                        scale = stats.sem(squared_errors)))

