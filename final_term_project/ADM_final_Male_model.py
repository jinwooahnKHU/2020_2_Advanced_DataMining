#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


raw_hn = pd.read_csv('hn_18_all_utf8.csv')


# In[3]:


raw_age_1 = raw_hn[raw_hn['age'] >= 13]


# In[4]:


raw_age_2 = raw_age_1[raw_age_1['age'] <= 18]


# In[5]:


raw_male_1 = raw_age_2[raw_age_2['sex'] == 1]


# In[6]:


#남녀통합
integ_sex_df = raw_age_2[['age','D_1_1','BO1','HE_BMI','BE5_1','BO2_1','BP1','BP6_10','BP6_2','BP6_31','BS10_3','BS9_2']]


# In[7]:


#남자만
raw_male = raw_male_1[['age','D_1_1','BO1','HE_BMI','BE5_1','BO2_1','BP1','BP6_10','BP6_2','BP6_31','BS10_3','BS9_2']]


# Male Model

# In[8]:


#na 행10 개라 그냥 떨굼
raw_male.dropna(inplace = True)


# In[9]:


#실수값들을 정수화
cols = ['age','D_1_1','BO1','HE_BMI','BE5_1','BO2_1','BP1','BP6_10','BP6_2','BP6_31','BS10_3','BS9_2']
for colname in cols:
    raw_male[cols] = raw_male[cols].astype(int)


# 남자 모델 범주화 작업

# In[10]:


raw_male.loc[raw_male['D_1_1'] == any([1,2,3]) , 'D_1_1'] = 1
raw_male.loc[raw_male['D_1_1'] == any([4,5]) , 'D_1_1'] = 2
raw_male.loc[raw_male['D_1_1'] == 9 , 'HE_BMI'] = 3


# In[11]:


raw_male.loc[raw_male['BO1'] == any([1,2]) , 'BO1'] = 1
raw_male.loc[raw_male['BO1'] == any([3]) , 'BO1'] = 2
raw_male.loc[raw_male['BO1'] == any([4,5]) , 'BO1'] = 3
raw_male.loc[raw_male['BO1'] == any([8,9]) , 'BO1'] = 4


# In[12]:


raw_male.loc[raw_male['HE_BMI'] < 25, 'HE_BMI'] = 0
raw_male.loc[raw_male['HE_BMI'] >= 25, 'HE_BMI'] = 1


# In[13]:


raw_male.loc[raw_male['BE5_1'] == any([1]) , 'BE5_1'] = 1
raw_male.loc[raw_male['BE5_1'] == any([2,3]) , 'BE5_1'] = 2
raw_male.loc[raw_male['BE5_1'] == any([4,5]) , 'BE5_1'] = 3
raw_male.loc[raw_male['BE5_1'] == any([8,9]) , 'BE5_1'] = 4


# In[14]:


raw_male.loc[raw_male['BO2_1'] == any([8,9]) , 'BO2_1'] = 5


# In[15]:


raw_male.loc[raw_male['BP1'] == any([1,2]) , 'BP1'] = 1
raw_male.loc[raw_male['BP1'] == any([3]) , 'BP1'] = 2
raw_male.loc[raw_male['BP1'] == any([4]) , 'BP1'] = 3
raw_male.loc[raw_male['BP1'] == any([8,9]) , 'BP1'] = 4


# In[16]:


raw_male.loc[raw_male['BS10_3'] == any([1]) , 'BS10_3'] = 1
raw_male.loc[raw_male['BS10_3'] == any([*range(2,20)]) , 'BS10_3'] = 2
raw_male.loc[raw_male['BS10_3'] == any([888,999]) , 'BS10_3'] = 4
raw_male.loc[raw_male['BS10_3'] >= 20 , 'BS10_3'] = 3


# In[17]:


raw_male.loc[raw_male['BS9_2'] == any([8,9]) , 'BS9_2'] = 4


# In[18]:


suicide = []
for i in range(raw_male.shape[0]):
    if 1 in list(raw_male[['BP6_10','BP6_2','BP6_31']].iloc[i]):
        suicide.append(1)
    else:
        suicide.append(0)
raw_male['suicide'] = suicide


# In[19]:


raw_male.drop(['BP6_10','BP6_2','BP6_31'], axis = 1, inplace = True)


# In[20]:


raw_male


# In[21]:


raw_male.index = [*range(208)]


# In[22]:


raw_male


# In[23]:


raw_male['HE_BMI'].value_counts()


# In[24]:


#X, y를 data, target이라는 변수로 설정
target = raw_male['HE_BMI']
data = raw_male.drop(['HE_BMI'], axis =1)


# In[25]:


#팩터화
cols = ['age','D_1_1','BO1','HE_BMI','BE5_1','BO2_1','BP1','BS10_3','BS9_2','suicide']
for colname in cols:
    raw_male[cols] = raw_male[cols].astype('category')


# In[26]:


from sklearn import svm


# 정규화 RBF Kernel Grid Search + K-fold = 5

# In[27]:


#정규화 + RBF kernel  K-fold = 5
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
#정규화
sc = StandardScaler()
sc.fit(data) 
data_tf = sc.transform(data)
from sklearn.model_selection import GridSearchCV
#Grid Search
C=[0.05,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,4,6,8,10]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=SVC(),param_grid=hyper,cv = 5, verbose=True)
gd.fit(data_tf,target)
print(gd.best_score_)
print(gd.best_estimator_)


# In[28]:


#위에서 나온 best parameter로 훈련하여 정확도, 리콜, 프리시전의 평균을 산출
svm_model = SVC(kernel='rbf', C=1, gamma=0.1)
svm_model.fit(data_tf, target)
scores = cross_val_score(svm_model, data_tf, target, cv=5)
scores_re = cross_val_score(svm_model, data_tf, target, cv=5, scoring='recall')
scores_pre = cross_val_score(svm_model, data_tf, target, cv=5, scoring='precision')
print("Accuracy mean : ", scores.mean())
print("Recall mean : ",scores_re.mean())
print("Precision mean : ",scores_pre.mean())


# 정규화 RBF Kernel Grid Search + K-fold = 10

# In[29]:


#정규화 + RBF kernel  K-fold = 10
#정규화
#Grid Search
C=[0.05,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,4,6,8,10]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=SVC(),param_grid=hyper,cv = 10, verbose=True)
gd.fit(data_tf,target)
print(gd.best_score_)
print(gd.best_estimator_)


# In[30]:


#위에서 나온 best parameter로 훈련하여 정확도, 리콜, 프리시전의 평균을 산출
svm_model = SVC(kernel='rbf', C=2, gamma=0.1)
svm_model.fit(data_tf, target)
scores = cross_val_score(svm_model, data_tf, target, cv=10)
scores_re = cross_val_score(svm_model, data_tf, target, cv=10, scoring='recall')
scores_pre = cross_val_score(svm_model, data_tf, target, cv=10, scoring='precision')
print("Accuracy mean : ", scores.mean())
print("Recall mean : ",scores_re.mean())
print("Precision mean : ",scores_pre.mean())


# 정규화 Linear Kernel Grid Search + K-fold = 5

# In[31]:


from matplotlib import pyplot as plt


# In[33]:


#정규화 + RBF kernel  K-fold = 5
#정규화
sc = StandardScaler()
sc.fit(data) 
data_tf = sc.transform(data)
from sklearn.model_selection import GridSearchCV
#Grid Search
C=[0.05,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,4,6,8,10]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=SVC(),param_grid=hyper,cv = 5, verbose=True)
gd.fit(data_tf,target)
print(gd.best_score_)
print(gd.best_estimator_)


# In[34]:


#위에서 나온 best parameter로 훈련하여 정확도, 리콜, 프리시전의 평균을 산출
svm_model = SVC(kernel='linear', C=0.1, gamma=0.1)
svm_model.fit(data_tf, target)
scores = cross_val_score(svm_model, data_tf, target, cv=5)
scores_re = cross_val_score(svm_model, data_tf, target, cv=5, scoring='recall')
scores_pre = cross_val_score(svm_model, data_tf, target, cv=5, scoring='precision')
print("Accuracy mean : ", scores.mean())
print("Recall mean : ",scores_re.mean())
print("Precision mean : ",scores_pre.mean())


# 정규화 Linear Kernel Grid Search + K-fold = 10

# In[35]:


#정규화 + RBF kernel  K-fold = 5
#정규화
sc = StandardScaler()
sc.fit(data) 
data_tf = sc.transform(data)
from sklearn.model_selection import GridSearchCV
#Grid Search
C=[0.05,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,4,6,8,10]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=SVC(),param_grid=hyper,cv = 10, verbose=True)
gd.fit(data_tf,target)
print(gd.best_score_)
print(gd.best_estimator_)


# In[36]:


#위에서 나온 best parameter로 훈련하여 정확도, 리콜, 프리시전의 평균을 산출
svm_model = SVC(kernel='linear', C=0.2, gamma=0.1)
svm_model.fit(data_tf, target)
scores = cross_val_score(svm_model, data_tf, target, cv=10)
scores_re = cross_val_score(svm_model, data_tf, target, cv=10, scoring='recall')
scores_pre = cross_val_score(svm_model, data_tf, target, cv=10, scoring='precision')
print("Accuracy mean : ", scores.mean())
print("Recall mean : ",scores_re.mean())
print("Precision mean : ",scores_pre.mean())


# 성별통합 모델 주요 변수 기여도 변수 기여도를 추출하는 것은 rbf에서는 안되고 linear에서만 가능하다.

# In[38]:


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


# In[39]:


# linear k = 5의 변수 기여도
features_names = ['age','D_1_1','BO1','HE_BMI','BE5_1','BO2_1','BP1','BS10_3','BS9_2','suicide']
svm_model = SVC(kernel='linear', C=0.1, gamma=0.1)
svm_model.fit(data_tf, target)
f_importances(abs(svm_model.coef_[0]), features_names)


# In[40]:


# linear k = 10의 변수 기여도
features_names = ['age','D_1_1','BO1','HE_BMI','BE5_1','BO2_1','BP1','BS10_3','BS9_2','suicide']
# c 값이 다름 
svm_model = SVC(kernel='linear', C=0.2, gamma=0.1)
svm_model.fit(data_tf, target)
f_importances(abs(svm_model.coef_[0]), features_names)

