Task4 掌握回归模型的评估及超参数调优

目的：虽然网格调参随机搜索方便又快速，但是人人都会用，竞赛中不易拉开差距。要想真正脱颖而出，理解模型的原理、掌握参数变化的趋势和优化方向才是硬核实力！
告别调参侠，真正拥有对模型的调试能力！


机器学习模型评估与超参数调优详解https://zhuanlan.zhihu.com/p/140040705

机器学习优化算法之贝叶斯优化 https://zhuanlan.zhihu.com/p/146329121 

最优化理论之无约束优化基本结构及其python应用 https://zhuanlan.zhihu.com/p/163405865

sklearn 中文文档  https://sklearn.apachecn.org/docs/master/50.html


1.回归模型的评估
   模型的评估包含三个指标：SSE(误差平方和)、R-square(决定系数)和Adjusted R-Square (校正决定系数）

1.1 SSE – 误差平方和
公式如下：

对同一个数据集，不同模型会有不同的SSE，SSE越小，说明模型的误差越小，准确率越高。
对不同的数据集，随着数据集的增加，误差也会增大，因此此时研究SSE没有意义。

1.2 R-square – 决定系数
决定系数是通过数据的变化来表征一个拟合的好坏。
公式如下：


分母是原始数据的离散程度，分子为预测数据和原始数据的误差平方和，二者相除可以消除原始数据离散程度的影响。

理论上R的取值范围（-∞，1]，但在实际应用中的取值范围为[0 1] ------ 实际操作中通常会选择拟合较好的曲线计算R²，因此很少出现-∞。
R越接近1，表明方程的变量对y的解释能力越强，这个模型对数据拟合的也较好。
R越接近0，表明模型拟合的越差。
经验值：>0.4， 拟合效果好。

1.3 Adjusted R-Square – 校正决定系数
公式如下：

其中n为样本数量，p为特征数量。
优点：校正决定系数消除了样本数量和特征数量的影响。

2.超参数调优
   2.1 参数调优
   模型参数是模型内部的配置变量，其值可以根据数据进行估计。一下是参数的一些特点：
   参数在预测中用到，是从数据估计中获取的。
   参数定义了可使用的模型，通常不由编程者手动设置。
   参数通常被保存为学习模型的一部分，它是机器学习算法的关键，通常由过去的训练数据中总结得出 。

2.2 超参数调优
模型超参数是模型外部的配置，其值无法从数据中估计。
超参数通常用于帮助估计模型参数，通常由人工指定。
超参数通常可以使用启发式设置。
超参数经常被调整为给定的预测建模问题，取不同的超参数的值对于模型的性能有不同的影响。

3.超参数的应用
   3.1 网格搜索GridSearchCV()
   网格搜索是从超参数空间中寻找最优的超参数，很像一个网格中找到一个最优的节点。
   举例： 𝜆=0.01,0.1,1.0 和 𝛼=0.01,0.1,1.0 , 组成一份排列组合，即：{[0.01,0.01],[0.01,0.1],[0.01,1],[0.1,0.01],[0.1,0.1],[0.1,1.0],[1,0.01],[1,0.1],[1,1]} ，然后针对每组超参数分别建立一个模型，然后选择测试误差最小的那组超参数。

3.2 随机搜索 RandomizedSearchCV()
随机搜索中的每个参数都是从可能的参数值的分布中采样的。
与网格搜索相比，随即搜索有以下优点：
（a）. 可以独立于参数数量和可能的值来选择计算成本。
（b）. 添加不影响性能的参数不会降低效率。

sklearn中对应函数
random_search =
RandomizedSearchCV(clf,param_distributions=param_dist,n_iter=n_iter_search, cv=5, iid=False)
具体含义参考：RandomizedSearchCV

2.2.2 代码实战
#随机搜索
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score #K折交叉验证

from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target

#分类问题，先建立一个分类器
clf = RandomForestClassifier(n_estimators=20)

#给定参数搜索范围
param_test={'max_depth':range(5,10,25),
           'max_features':sp_randint(1,11),
           "min_samples_split": sp_randint(2, 11)}

#RandomSearch+CV选取超参数
random_search = RandomizedSearchCV(clf,param_distributions =param_test,n_iter=20,cv=5)

random_search.fit(X,y)
print("随机搜索最优得分：",random_search.best_score_)
print("随机搜索最优参数组合：\n",random_search.best_params_)



随机搜索最优得分： 0.8820458062519343
随机搜索最优参数组合：
 {'max_depth': 5, 'max_features': 7, 'min_samples_split': 5}

根据以上不同模型的代码及运行结果可以看出来，随机搜索的优势体现在能对一个给定范围内的参数求最优。
