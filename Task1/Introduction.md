## Task01：熟悉机器学习的三大主要任务

![机器学习图](https://user-images.githubusercontent.com/62379948/111166526-23ce8080-85db-11eb-9d0d-e8f73fce6e3e.jpg)

顶楼放出萌佬的  [code viewer address](https://nbviewer.jupyter.org/github/datawhalechina/team-learning-data-mining/blob/master/EnsembleLearning/CH2-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B%E5%9B%9E%E9%A1%BE/%E7%AC%AC%E4%BA%8C%E7%AB%A0%EF%BC%9A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80.ipynb) 

基于萌佬的code与思路，再结合自己的理解和尝试进行总结；
数学方面还会持续奋斗

[pumpkin-book](https://datawhalechina.github.io/pumpkin-book)
西瓜书/《机器学习》 周志华
joyful-pandas

1.机器学习的定义
我们如何理解机器学习？
计算机中的“数据”对应人的“经验”，让计算机来学习这些经验数据，生成一个算法模型，在面对新的情况中，计算机便能作出有效的判断。
理解数据离不开数学模型，通过数学方式的推演，发现数据中的规律，用以对数据的分析和预测。
数据通常由一组向量组成，这组向量中的每个向量都是一个样本，我们用Xi来表示一个样本，其中i=1,2,3,...,N,共N个样本，每个样本Xi=(Xi1,Xi2,...,Xip,Yi)共p+1个维度，前p个维度的每个维度我们称为一个特征，最后一个维度Yi我们称为因变量(响应变量)。特征用来描述影响因变量的因素，如：我们要探寻身高是否会影响体重的关系的时候，身高就是一个特征，体重就是一个因变量。通常在一个数据表dataframe里面，一行表示一个样本Xi，一列表示一个特征。
根据数据是否有因变量，机器学习的任务可分为：有监督学习和无监督学习。
教科书《机器学习：一种人工智能方法》作者CMU教授Tom Mitchell给出了一个形式化的定义，假设：

P：计算机程序在某任务类T上的性能。
T：计算机程序希望实现的任务类。
E：表示经验，即历史的数据集。
若该计算机程序通过利用经验E在任务T上获得了性能P的改善，则称该程序对E进行了学习。



Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
![机器学习图](https://user-images.githubusercontent.com/62379948/111166503-1f09cc80-85db-11eb-9c0d-5837d0b02486.jpg)
