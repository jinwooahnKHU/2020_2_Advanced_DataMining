#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version = 1)
mnist.keys()


# In[2]:


X, y = mnist['data'], mnist['target']
X.shape


# In[3]:


y.shape


# In[5]:


import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape((28,28))

plt.imshow(some_digit_image, cmap = 'binary')
plt.axis("off")
plt.show()


# In[6]:


y[0]


# In[8]:


import numpy as np 
y = y.astype(np.uint8)


# In[9]:


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[10]:


y_train_5 = (y_train==5)
y_test_5 = (y_test == 5)


# In[11]:


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])


# In[12]:


from sklearn.model_selection import cross_val_score
#cv : 3개의 fold로 나누겠다~
#이건 지금 훈련 데이터를 사용해서 검증한거임 목적에 따라 
#모델에 들어갈 하이퍼 파라미터들을 튜닝하는 용으로도 사용할 수도 있고
#실제 모델 이후 evaluation에 사용할 수도 있음!
cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = "accuracy")
#나온 값을 평균내서 대푯값으로 사용가능

