
## Task02：掌握基本的回归模型

一般来说，一个完整的机器学习项目分为以下步骤：
明确项目任务：回归/分类
收集数据集并选择合适的特征。
选择度量模型性能的指标。
选择具体的模型并进行训练以优化模型。
评估模型的性能并调参。



预测建模就是使用历史数据建立一个模型，去给没有答案的新数据做预测的问题。

而预测建模可以被描述成一个近似求取从输入变量（X）到输出变量（y）的映射函数的数学问题。这被称为函数逼近问题。



建模算法的任务就是在给定的可用时间和资源的限制下，去寻找最佳映射函数。

一般而言，我们可以将函数逼近任务划分为分类任务和回归任务。
我们先介绍回归任务
回归预测建模是逼近一个从输入变量（X）到连续的输出变量（y）的函数映射。
因为回归预测问题预测的是一个数量，所以模型的性能可以用预测结果中的错误来评价。

有很多评价回归预测模型的方式，但是最常用的一个可能是计算误差值的均方根，即 RMSE。
例如，如果回归预测模型做出了两个预测结果，一个是 1.5，对应的期望结果是 1.0；另一个是 3.3 对应的期望结果是 3.0. 那么，这两个回归预测的 RMSE 如下：
`RMSE = sqrt(average(error^2))`

`RMSE = sqrt(((1.0 - 1.5)^2 + (3.0 - 3.3)^2) / 2)`

`RMSE = sqrt((0.25 + 0.09) / 2)`

`RMSE = sqrt(0.17)`

`RMSE = 0.412`

使用 RMSE 的好处就是错误评分的单位与预测结果是一样的。
一个能够学习回归预测模型的算法称作回归算法。

分类是预测一个离散标签的任务


以下列出一些回归的通用方法，
对于CH2的资料另开分析
对于算法详细介绍后续补充


**1、线性回归**

线性回归拟合一个带系数的线性模型，以最小化数据中的观测值与线性预测值之间的残差平方和。

sklearn中也存在线性回归的算法库的接口，代码示例如下所示：


```
#加载线性模型算法库
from sklearn import linear_model
# 创建线性回归模型的对象
regr = linear_model.LinearRegression()
# 利用训练集训练线性模型
regr.fit(X_train, y_train)
# 使用测试集做预测
y_pred = regr.predict(X_test)
```



**2、岭回归**

上述的线性回归算法使用最小二乘法优化各个系数，对于岭回归来说，岭回归通过对系数进行惩罚(L2范式)来解决普通最小二乘法的一些问题，例如，当特征之间完全共线性(有解)或者说特征之间高度相关，这个时候适合用岭回归。



```
#加载线性模型算法库
from sklearn.linear_model import Ridge
# 创建岭回归模型的对象
reg = Ridge(alpha=.5)
# 利用训练集训练岭回归模型
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
#输出各个系数
reg.coef_
reg.intercept_ 
```



**3、Lasso回归**

Lasso是一个估计稀疏稀疏的线性模型。它在某些情况下很有用，由于它倾向于选择参数值较少的解，有效地减少了给定解所依赖的变量的数量。Lasso模型在最小二乘法的基础上加入L1范式作为惩罚项。



```
#加载Lasso模型算法库
from sklearn.linear_model import Lasso
# 创建Lasso回归模型的对象
reg = Lasso(alpha=0.1)
# 利用训练集训练Lasso回归模型
reg.fit([[0, 0], [1, 1]], [0, 1])
"""
Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
"""
# 使用测试集做预测
reg.predict([[1, 1]])
```



**4、Elastic Net回归**

Elastic Net 是一个线性模型利用L1范式和L2范式共同作为惩罚项。这种组合既可以学习稀疏模型，同时可以保持岭回归的正则化属性。



```
#加载ElasticNet模型算法库
from sklearn.linear_model import ElasticNet
#加载数据集
from sklearn.datasets import make_regression
X, y = make_regression(n_features=2, random_state=0)
#创建ElasticNet回归模型的对象
regr = ElasticNet(random_state=0)
# 利用训练集训练ElasticNet回归模型
regr.fit(X, y)
print(regr.coef_) 
print(regr.intercept_) 
print(regr.predict([[0, 0]])) 
```



**5、贝叶斯岭回归**

贝叶斯岭回归模型和岭回归类似。贝叶斯岭回归通过最大化边际对数似然来估计参数。



```
from sklearn.linear_model import BayesianRidge
X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
Y = [0., 1., 2., 3.]
reg = BayesianRidge()
reg.fit(X, Y)
```



**6、SGD回归**

上述的线性模型通过最小二乘法来优化损失函数，SGD回归也是一种线性回归，不同的是，它通过随机梯度下降最小化正则化经验损失。



```
import numpy as np
from sklearn import linear_model
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
clf.fit(X, y)
"""
SGDRegressor(alpha=0.0001, average=False, early_stopping=False,
       epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
       learning_rate='invscaling', loss='squared_loss', max_iter=1000,
       n_iter=None, n_iter_no_change=5, penalty='l2', power_t=0.25,
       random_state=None, shuffle=True, tol=0.001, validation_fraction=0.1,
       verbose=0, warm_start=False)
"""
```





**7、SVR**

众所周知，支持向量机在分类领域应用非常广泛，支持向量机的分类方法可以被推广到解决回归问题，这个就称为支持向量回归。支持向量回归算法生成的模型同样地只依赖训练数据集中的一个子集(和支持向量分类算法类似)。



```
#加载SVR模型算法库
from sklearn.svm import SVR
#训练集
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
#创建SVR回归模型的对象
clf = SVR()
# 利用训练集训练SVR回归模型
clf.fit(X, y) 
"""
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)
"""
clf.predict([[1, 1]])
```





**8、KNN回归**

在数据标签是连续变量而不是离散变量的情况下，可以使用KNN回归。分配给查询点的标签是根据其最近邻居标签的平均值计算的。



```
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, y) 
print(neigh.predict([[1.5]]))
```





**9、决策树回归**

决策树也可以应用于回归问题，使用sklearn的DecisionTreeRegressor类。



```
from sklearn.tree import  DecisionTreeRegressor 
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])
```



**10、神经网络**

神经网络使用slearn中MLPRegressor类实现了一个多层感知器(MLP)，它使用在输出层中没有激活函数的反向传播进行训练，也可以将衡等函数视为激活函数。因此，它使用平方误差作为损失函数，输出是一组连续的值。



```
from sklearn.neural_network import MLPRegressor
mlp=MLPRegressor()
mlp.fit(X_train,y_train)
"""
MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
"""
y_pred = mlp.predict(X_test)
```



**11.RandomForest回归**

RamdomForest回归也是一种经典的集成算法之一。



```
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0,
                             n_estimators=100)
regr.fit(X, y)
print(regr.feature_importances_)
print(regr.predict([[0, 0, 0, 0]]))
```



**11、XGBoost回归**

XGBoost近些年在学术界取得的成果连连捷报，基本所有的机器学习比赛的冠军方案都使用了XGBoost算法，对于XGBoost的算法接口有两种，这里我仅介绍XGBoost的sklearn接口。

更多请参考： 

https://xgboost.readthedocs.io/en/latest/python/index.html



```
import xgboost as xgb
xgb_model = xgb.XGBRegressor(max_depth = 3,
                             learning_rate = 0.1,
                             n_estimators = 100,
                             objective = 'reg:linear',
                             n_jobs = -1)

xgb_model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)], 
              eval_metric='logloss',
              verbose=100)
y_pred = xgb_model.predict(X_test)
print(mean_squared_error(y_test, y_pred))
```





**12、LightGBM回归**

LightGBM作为另一个使用基于树的学习算法的梯度增强框架。在算法竞赛也是每逢必用的神器，且要想在竞赛取得好成绩，LightGBM是一个不可或缺的神器。相比于XGBoost，LightGBM有如下优点，训练速度更快，效率更高效；低内存的使用量。对于LightGBM的算法接口有两种，这里我同样介绍LightGBM的sklearn接口。

更多请参考：https://lightgbm.readthedocs.io/en/latest/

```
import lightgbm as lgb
gbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(X_train, y_train,
        eval_set=[(X_train, y_train)], 
        eval_metric='logloss',
        verbose=100)
y_pred = gbm.predict(X_test)
print(mean_squared_error(y_test, y_pred))
```
