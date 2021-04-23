
机器学习大杀器——梯度提升树GBDT https://zhuanlan.zhihu.com/p/45145899

Adaboost：https://link.medium.com/udXDrLndAfb （感谢@你队-hyx同学的链接分享和@一路向北-黄元帅 的群内pdf分享）

清华大学【数据挖掘：集成学习】：https://www.bilibili.com/video/BV1mt411a7FT?p=3&spm_id_from=333.788.b_6d756c74695f70616765.3 

# 前向分步算法

回看Adaboost的算法内容，我们需要通过计算M个基本分类器，每个分类器的错误率、样本权重以及模型权重。我们可以认为：Adaboost每次学习单一分类器以及单一分类器的参数(权重)。接下来，我们抽象出Adaboost算法的整体框架逻辑，构建集成学习的一个非常重要的框架----前向分步算法，有了这个框架，我们不仅可以解决分类问题，也可以解决回归问题。
**(1) 加法模型：**
在Adaboost模型中，我们把每个基本分类器合成一个复杂分类器的方法是每个基本分类器的加权和，即：f(x)=∑Mm=1βmb(x;γm)f(x)=∑m=1Mβmb(x;γm)，其中，b(x;γm)b(x;γm)为即基本分类器，γmγm为基本分类器的参数，βmβm为基本分类器的权重，显然这与第二章所学的加法模型。为什么这么说呢？大家把b(x;γm)b(x;γm)看成是即函数即可。
在给定训练数据以及损失函数L(y,f(x))L(y,f(x))的条件下，学习加法模型f(x)f(x)就是：

minβm,γmN∑i=1L(yi,M∑m=1βmb(xi;γm))minβm,γm∑i=1NL(yi,∑m=1Mβmb(xi;γm))

通常这是一个复杂的优化问题，很难通过简单的凸优化的相关知识进行解决。前向分步算法可以用来求解这种方式的问题，它的基本思路是：因为学习的是加法模型，如果从前向后，每一步只优化一个基函数及其系数，逐步逼近目标函数，那么就可以降低优化的复杂度。具体而言，每一步只需要优化：

minβ,γN∑i=1L(yi,βb(xi;γ))minβ,γ∑i=1NL(yi,βb(xi;γ))

**(2) 前向分步算法：**
给定数据集T={(x1,y1),(x2,y2),⋯,(xN,yN)}T={(x1,y1),(x2,y2),⋯,(xN,yN)}，xi∈X⊆Rnxi∈X⊆Rn，yi∈Y={+1,−1}yi∈Y={+1,−1}。损失函数L(y,f(x))L(y,f(x))，基函数集合{b(x;γ)}{b(x;γ)}，我们需要输出加法模型f(x)f(x)。

- 初始化：f0(x)=0f0(x)=0
- 对m = 1,2,...,M:
  - (a) 极小化损失函数：(βm,γm)=argminβ,γN∑i=1L(yi,fm−1(xi)+βb(xi;γ))(βm,γm)=arg⁡minβ,γ∑i=1NL(yi,fm−1(xi)+βb(xi;γ))得到参数βmβm与γmγm
  - (b) 更新：fm(x)=fm−1(x)+βmb(x;γm)fm(x)=fm−1(x)+βmb(x;γm)
- 得到加法模型：f(x)=fM(x)=M∑m=1βmb(x;γm)f(x)=fM(x)=∑m=1Mβmb(x;γm)

这样，前向分步算法将同时求解从m=1到M的所有参数βmβm，γmγm的优化问题简化为逐次求解各个βmβm，γmγm的问题。
**(3) 前向分步算法与Adaboost的关系：**
由于这里不是我们的重点，我们主要阐述这里的结论，不做相关证明，具体的证明见李航老师的《统计学习方法》第八章的3.2节。Adaboost算法是前向分步算法的特例，Adaboost算法是由基本分类器组成的加法模型，损失函数为指数损失函数。





# 5. 梯度提升决策树(GBDT)





(1) 基于残差学习的提升树算法：
在前面的学习过程中，我们一直讨论的都是分类树，比如Adaboost算法，并没有涉及回归的例子。在上一小节我们提到了一个加法模型+前向分步算法的框架，那能否使用这个框架解决回归的例子呢？答案是肯定的。接下来我们来探讨下如何使用加法模型+前向分步算法的框架实现回归问题。
在使用加法模型+前向分步算法的框架解决问题之前，我们需要首先确定框架内使用的基函数是什么，在这里我们使用决策树分类器。前面第二章我们已经学过了回归树的基本原理，树算法最重要是寻找最佳的划分点，分类树用纯度来判断最佳划分点使用信息增益（ID3算法），信息增益比（C4.5算法），基尼系数（CART分类树）。但是在回归树中的样本标签是连续数值，可划分点包含了所有特征的所有可取的值。所以再使用熵之类的指标不再合适，取而代之的是平方误差，它能很好的评判拟合程度。基函数确定了以后，我们需要确定每次提升的标准是什么。回想Adaboost算法，在Adaboost算法内使用了分类错误率修正样本权重以及计算每个基本分类器的权重，那回归问题没有分类错误率可言，也就没办法在这里的回归问题使用了，因此我们需要另辟蹊径。模仿分类错误率，我们用每个样本的残差表示每次使用基函数预测时没有解决的那部分问题。因此，我们可以得出如下算法：
输入数据集T={(x1,y1),(x2,y2),⋯,(xN,yN)},xi∈X⊆Rn,yi∈Y⊆RT={(x1,y1),(x2,y2),⋯,(xN,yN)},xi∈X⊆Rn,yi∈Y⊆R，输出最终的提升树fM(x)fM(x)

- 初始化f0(x)=0f0(x)=0
- 对m = 1,2,...,M：
  - 计算每个样本的残差:rmi=yi−fm−1(xi),i=1,2,⋯,Nrmi=yi−fm−1(xi),i=1,2,⋯,N
  - 拟合残差rmirmi学习一棵回归树，得到T(x;Θm)T(x;Θm)
  - 更新fm(x)=fm−1(x)+T(x;Θm)fm(x)=fm−1(x)+T(x;Θm)
- 得到最终的回归问题的提升树：fM(x)=∑Mm=1T(x;Θm)fM(x)=∑m=1MT(x;Θm)

下面我们用一个实际的案例来使用这个算法：(案例来源：李航老师《统计学习方法》)
训练数据如下表，学习这个回归问题的提升树模型，考虑只用树桩作为基函数。

至此，我们已经能够建立起依靠加法模型+前向分步算法的框架解决回归问题的算法，叫提升树算法。那么，这个算法还是否有提升的空间呢？


(2) 梯度提升决策树算法(GBDT)：

提升树利用加法模型和前向分步算法实现学习的过程，当损失函数为平方损失和指数损失时，每一步优化是相当简单的，也就是我们前面探讨的提升树算法和Adaboost算法。但是对于一般的损失函数而言，往往每一步的优化不是那么容易，针对这一问题，我们得分析问题的本质，也就是是什么导致了在一般损失函数条件下的学习困难。对比以下损失函数：

 Setting Loss Function −∂L(yi,f(xi))/∂f(xi) Regression 12[yi−f(xi)]2yi−f(xi) Regression |yi−f(xi)|sign[yi−f(xi)] Regression Huber yi−f(xi) for |yi−f(xi)|≤δmδmsign[yi−f(xi)] for |yi−f(xi)|>δm where δm=α th-quantile {|yi−f(xi)|} Classification Deviance k th component: I(yi=Gk)−pk(xi) Setting Loss Function −∂L(yi,f(xi))/∂f(xi) Regression 12[yi−f(xi)]2yi−f(xi) Regression |yi−f(xi)|sign⁡[yi−f(xi)] Regression Huber yi−f(xi) for |yi−f(xi)|≤δmδmsign⁡[yi−f(xi)] for |yi−f(xi)|>δm where δm=α th-quantile {|yi−f(xi)|} Classification Deviance k th component: I(yi=Gk)−pk(xi)

观察Huber损失函数：

Lδ(y,f(x))={12(y−f(x))2 for |y−f(x)|≤δδ|y−f(x)|−12δ2 otherwise Lδ(y,f(x))={12(y−f(x))2 for |y−f(x)|≤δδ|y−f(x)|−12δ2 otherwise 

针对上面的问题，Freidman提出了梯度提升算法(gradient boosting)，这是利用最速下降法的近似方法，利用损失函数的负梯度在当前模型的值−[∂L(y,f(xi))∂f(xi)]f(x)=fm−1(x)−[∂L(y,f(xi))∂f(xi)]f(x)=fm−1(x)作为回归问题提升树算法中的残差的近似值，拟合回归树。**与其说负梯度作为残差的近似值，不如说残差是负梯度的一种特例。**
以下开始具体介绍梯度提升算法：
输入训练数据集T={(x1,y1),(x2,y2),⋯,(xN,yN)},xi∈X⊆Rn,yi∈Y⊆RT={(x1,y1),(x2,y2),⋯,(xN,yN)},xi∈X⊆Rn,yi∈Y⊆R和损失函数L(y,f(x))L(y,f(x))，输出回归树^f(x)f^(x)

- 初始化f0(x)=argminc∑Ni=1L(yi,c)f0(x)=arg⁡minc∑i=1NL(yi,c)
- 对于m=1,2,...,M：
  - 对i = 1,2,...,N计算：rmi=−[∂L(yi,f(xi))∂f(xi)]f(x)=fm−1(x)rmi=−[∂L(yi,f(xi))∂f(xi)]f(x)=fm−1(x)
  - 对rmirmi拟合一个回归树，得到第m棵树的叶结点区域Rmj,j=1,2,⋯,JRmj,j=1,2,⋯,J
  - 对j=1,2,...J，计算：cmj=argminc∑xi∈RmjL(yi,fm−1(xi)+c)cmj=arg⁡minc∑xi∈RmjL(yi,fm−1(xi)+c)
  - 更新fm(x)=fm−1(x)+∑Jj=1cmjI(x∈Rmj)fm(x)=fm−1(x)+∑j=1JcmjI(x∈Rmj)
- 得到回归树：^f(x)=fM(x)=∑Mm=1∑Jj=1cmjI(x∈Rmj)f^(x)=fM(x)=∑m=1M∑j=1JcmjI(x∈Rmj)

下面，我们来使用一个具体的案例来说明GBDT是如何运作的(案例来源：https://blog.csdn.net/zpalyq110/article/details/79527653 )：
下面的表格是数据：
![jupyter](vscode-webview-resource://87cb26c1-bba7-4ba6-b3cc-3813acf674f9/file///Users/rachael-y/Desktop/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/CH4-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B9%8Bboosting/6.png)
学习率：learning_rate=0.1，迭代次数：n_trees=5，树的深度：max_depth=3
平方损失的负梯度为：

−[∂L(y,f(xi)))∂f(xi)]f(x)=ft−1(x)=y−f(xi)−[∂L(y,f(xi)))∂f(xi)]f(x)=ft−1(x)=y−f(xi)

c=(1.1+1.3+1.7+1.8)/4=1.475，f0(x)=c=1.475c=(1.1+1.3+1.7+1.8)/4=1.475，f0(x)=c=1.475
![jupyter](vscode-webview-resource://87cb26c1-bba7-4ba6-b3cc-3813acf674f9/file///Users/rachael-y/Desktop/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/CH4-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B9%8Bboosting/8.png)
学习决策树，分裂结点：
![jupyter](vscode-webview-resource://87cb26c1-bba7-4ba6-b3cc-3813acf674f9/file///Users/rachael-y/Desktop/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/CH4-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B9%8Bboosting/9.png)
![jupyter](vscode-webview-resource://87cb26c1-bba7-4ba6-b3cc-3813acf674f9/file///Users/rachael-y/Desktop/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/CH4-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B9%8Bboosting/10.png)
对于左节点，只有0，1两个样本，那么根据下表我们选择年龄7进行划分：
![jupyter](vscode-webview-resource://87cb26c1-bba7-4ba6-b3cc-3813acf674f9/file///Users/rachael-y/Desktop/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/CH4-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B9%8Bboosting/11.png)
对于右节点，只有2，3两个样本，那么根据下表我们选择年龄30进行划分：
![jupyter](vscode-webview-resource://87cb26c1-bba7-4ba6-b3cc-3813acf674f9/file///Users/rachael-y/Desktop/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/CH4-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B9%8Bboosting/12.png)
![jupyter](vscode-webview-resource://87cb26c1-bba7-4ba6-b3cc-3813acf674f9/file///Users/rachael-y/Desktop/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/CH4-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B9%8Bboosting/13.png)

因此根据Υj1=argminΥ∑xi∈Rj1L(yi,f0(xi)+Υ)Υj1=arg⁡min⏟Υ∑xi∈Rj1L(yi,f0(xi)+Υ)：

(x0∈R11),Υ11=−0.375(x1∈R21),Υ21=−0.175(x2∈R31),Υ31=0.225(x3∈R41),Υ41=0.325(x0∈R11),Υ11=−0.375(x1∈R21),Υ21=−0.175(x2∈R31),Υ31=0.225(x3∈R41),Υ41=0.325

这里其实和上面初始化学习器是一个道理，平方损失，求导，令导数等于零，化简之后得到每个叶子节点的参数ΥΥ,其实就是标签值的均值。 最后得到五轮迭代：
![jupyter](vscode-webview-resource://87cb26c1-bba7-4ba6-b3cc-3813acf674f9/file///Users/rachael-y/Desktop/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/CH4-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B9%8Bboosting/14.png)
最后的强学习器为：f(x)=f5(x)=f0(x)+∑5m=1∑4j=1ΥjmI(x∈Rjm)f(x)=f5(x)=f0(x)+∑m=15∑j=14ΥjmI(x∈Rjm)。
其中：

f0(x)=1.475f2(x)=0.0205f3(x)=0.1823f4(x)=0.1640f5(x)=0.1476f0(x)=1.475f2(x)=0.0205f3(x)=0.1823f4(x)=0.1640f5(x)=0.1476

预测结果为：

f(x)=1.475+0.1∗(0.2250+0.2025+0.1823+0.164+0.1476)=1.56714f(x)=1.475+0.1∗(0.2250+0.2025+0.1823+0.164+0.1476)=1.56714

为什么要用学习率呢？这是Shrinkage的思想，如果每次都全部加上（学习率为1）很容易一步学到位导致过拟合。







下面我们来使用sklearn来使用GBDT：
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gra#sklearn.ensemble.GradientBoostingClassifier




from sklearn.metrics import mean_squared_error

from sklearn.datasets import make_friedman1

from sklearn.ensemble import GradientBoostingRegressor

 

'''

GradientBoostingRegressor参数解释：

loss：{‘ls’, ‘lad’, ‘huber’, ‘quantile’}, default=’ls’：‘ls’ 指最小二乘回归. ‘lad’ (最小绝对偏差) 是仅

基于输入变量的顺序信息的高度鲁棒的损失函数。. ‘huber’ 是两者的结合. ‘quantile’允许分位数回归（用于alpha指定分位

数）

learning_rate：学习率缩小了每棵树的贡献learning_rate。在learning_rate和n_estimators之间需要权衡。

n_estimators：要执行的提升次数。

subsample：用于拟合各个基础学习者的样本比例。如果小于1.0，则将导致随机梯度增强。subsample与参数n_estimators。

选择会导致方差减少和偏差增加。subsample < 1.0

criterion：{'friedman_mse'，'mse'，'mae'}，默认='friedman_mse'：“ mse”是均方误差，“ mae”是平均绝对误差。

默认值“ friedman_mse”通常是最好的，因为在某些情况下它可以提供更好的近似值。

min_samples_split：拆分内部节点所需的最少样本数

min_samples_leaf：在叶节点处需要的最小样本数。

min_weight_fraction_leaf：在所有叶节点处（所有输入样本）的权重总和中的最小加权分数。如果未提供sample_weight，

则样本的权重相等。

max_depth：各个回归模型的最大深度。最大深度限制了树中节点的数量。调整此参数以获得最佳性能；最佳值取决于输入变量的

相互作用。

min_impurity_decrease：如果节点分裂会导致杂质的减少大于或等于该值，则该节点将被分裂。

min_impurity_split：提前停止树木生长的阈值。如果节点的杂质高于阈值，则该节点将分裂

max_features{‘auto’, ‘sqrt’, ‘log2’}，int或float：寻找最佳分割时要考虑的功能数量：

 

如果为int，则max_features在每个分割处考虑特征。

 

如果为float，max_features则为小数，并 在每次拆分时考虑要素。int(max_features * n_features)

 

如果“auto”，则max_features=n_features。

 

如果是“ sqrt”，则max_features=sqrt(n_features)。

 

如果为“ log2”，则为max_features=log2(n_features)。

 

如果没有，则max_features=n_features。

'''

 

X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)

X_train, X_test = X[:200], X[200:]

y_train, y_test = y[:200], y[200:]

est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,

  max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)

mean_squared_error(y_test, est.predict(X_test))

```
5.009154859960321
```




from sklearn.datasets import make_regression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

X, y = make_regression(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(

  X, y, random_state=0)

reg = GradientBoostingRegressor(random_state=0)

reg.fit(X_train, y_train)

reg.score(X_test, y_test)

 

```
0.43848663277068134
```









这里给大家一个小作业，就是大家总结下GradientBoostingRegressor与GradientBoostingClassifier函数的各个参数的意思！参考文档：
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gra#sklearn.ensemble.GradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gra#sklearn.ensemble.GradientBoostingClassifier)

