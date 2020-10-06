#!/usr/bin/env python
# coding: utf-8

# *** Training DecisionTreeClassifier with iris Data Set ***

# In[1]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


# In[43]:


import numpy as np
import pandas as pd


# In[28]:


import os


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[31]:


# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# In[3]:


iris= load_iris()


# In[24]:


#꽃잎의 길이와 너비를 X로 정의
X = iris.data[:,2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth = 2, random_state = 42)
tree_clf.fit(X, y)


# In[34]:


#결정 트리를 시각화
from sklearn.tree import export_graphviz
from graphviz import Source
export_graphviz(
        tree_clf,
        out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))


# ==> Node에서 Samples 속성은 얼마나 많은 훈련 샘플이 적용되었는지를 나타내어줌.
# Gini는 불순도를 측정하여 표시한 것.

# * 클래스 확률을 기반으로 하는 추정 *

# In[37]:


tree_clf.predict_proba([[5,1.5]])


# In[39]:


tree_clf.predict([[5,1.5]]) #위의 셀에서 가장 확률이 높은 클래스를 반환


# *** 결정 트리를 회귀에 사용 ***

# 1. 데이터 셋 만들기
# y = 4*(x - 0.5)^2 을 사용하였ㅇ며 y값에 랜덤한 잡음을 넣어보자

# In[47]:


# 2차식으로 만든 데이터셋 + 잡음
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)   #표준정규분포에서 난수를 발생시켜 크기 (m,1)의 매트릭스에 할당한다.
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10  # 무작위 잡음 추가


# In[48]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth = 2, random_state = 42)
tree_reg.fit(X,y)


# In[49]:


tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_regression_predictions(tree_reg1, X, y)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, "Depth=0", fontsize=15)
plt.text(0.01, 0.2, "Depth=1", fontsize=13)
plt.text(0.65, 0.8, "Depth=1", fontsize=13)
plt.legend(loc="upper center", fontsize=18)
plt.title("max_depth=2", fontsize=14)

plt.sca(axes[1])
plot_regression_predictions(tree_reg2, X, y, ylabel=None)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
    plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
plt.text(0.3, 0.5, "Depth=2", fontsize=13)
plt.title("max_depth=3", fontsize=14)

save_fig("tree_regression_plot")
plt.show()


# In[50]:


export_graphviz(
        tree_reg1,
        out_file=os.path.join(IMAGES_PATH, "regression_tree.dot"),
        feature_names=["x1"],
        rounded=True,
        filled=True
    )


# In[51]:


Source.from_file(os.path.join(IMAGES_PATH, "regression_tree.dot"))

