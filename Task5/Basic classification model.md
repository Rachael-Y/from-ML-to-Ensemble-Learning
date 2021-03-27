Task5 掌握基本的分类模型

简单理解线性判别分析https://zhuanlan.zhihu.com/p/66088884

Python实现决策树https://zhuanlan.zhihu.com/p/65304798

数学估计方法——最小方差、极大似然与贝叶斯https://zhuanlan.zhihu.com/p/41015902

机器学习之sklearn基本分类方法https://zhuanlan.zhihu.com/p/173945775

机器学习优化算法之贝叶斯优化https://zhuanlan.zhihu.com/p/146329121

机器学习之简单分类模型https://zhuanlan.zhihu.com/p/137215803

小雨姑娘的机器学习笔记https://www.zhihu.com/column/mlbasic



(1) 收集数据集并选择合适的特征;

(2) 选择度量模型性能的指标;
![77b6c4ea225ff8ffbf4d5e9837bc95c](https://user-images.githubusercontent.com/62379948/112726312-b1a15880-8f57-11eb-9148-060a50cfc567.png)


分类模型的指标：
准确率：分类正确的样本数占总样本的比例
精度：预测为正且分类正确的样本占预测值为正的比例 
召回率：预测为正且分类正确的样本占类别为正的比例 
F1值：综合衡量精度和召回率
ROC曲线：以假阳率为横轴，真阳率为纵轴画出来的曲线，曲线下方面积越大越好。

![8d755e77bb51710274b1b9f8e5bf7fc](https://user-images.githubusercontent.com/62379948/112726354-e9100500-8f57-11eb-9e87-60a34ac745f3.png)

https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
(https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)


