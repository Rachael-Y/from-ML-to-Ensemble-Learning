links:https://mp.weixin.qq.com/s/Pcr7oAsA0BOQ_G0xqukSoA

Boosting 是使用集成学习概念的技术之一。 Boosting 结合了多个简单模型（也称为弱学习者或基本估计量）来生成最终输出。

机器学习中的4种 Boosting 


梯度提升机（GBM）
极端梯度提升机（XGBM）
轻量梯度提升机（LightGBM）
分类提升（CatBoost）

1、梯度提升机（GBM）

梯度提升机（GBM）结合了来自多个决策树的预测来生成最终预测。注意，梯度提升机中的所有弱学习者都是决策树。

但是，如果我们使用相同的算法，那么使用一百个决策树比使用单个决策树好吗？不同的决策树如何从数据中捕获不同的信号/信息呢？

这就是窍门––每个决策树中的节点采用不同的功能子集来选择最佳拆分。这意味着各个树并不完全相同，因此它们能够从数据中捕获不同的信号。

另外，每棵新树都考虑到先前树所犯的错误。因此，每个连续的决策树都是基于先前树的错误。这就是按顺序构建梯度 Boosting 中树的方式。



2、极端梯度提升机（XGBM）

极端梯度提升机（XGBoost）是另一种流行的 Boosting 。实际上，XGBoost只是GBM算法的改进版！XGBoost的工作过程与GBM相同。XGBoost中的树是按顺序构建的尝试用于更正先前树的错误。
 
但是， XGBoost某些功能稍微优于GBM：

1）最重要的一点是XGBM实现了并行预处理（在节点级别），这使其比GBM更快。

2）XGBoost还包括各种正则化技术，可减少过度拟合并改善整体表现。你可以通过设置XGBoost算法的超参数来选择正则化技术。

此外，如果使用的是XGBM算法，则不必担心会在数据集中插入缺失值。XGBM模型可以自行处理缺失值。在训练过程中，模型将学习缺失值是在右节点还是左节点中。

3、轻量梯度提升机（LightGBM）

由于其速度和效率，LightGBM  Boosting 如今变得越来越流行。LightGBM能够轻松处理大量数据。但是请注意，该算法在少数数据点上的性能不佳。

让我们花点时间来了解为什么会出现这种情况。

LightGBM中的树具有叶向生长的，而不是水平生长的。在第一次分割之后，下一次分割仅在损失较大的叶节点上进行。

考虑下图所示的示例：



第一次分割后，左侧节点的损耗较高，因此被选择用于下一个分割。现在，我们有三个叶节点，而中间叶节点的损耗最高。LightGBM算法的按叶分割使它能够处理大型数据集。

为了加快训练过程，LightGBM使用基于直方图的方法来选择最佳分割。对于任何连续变量而不是使用各个值，这些变量将被分成仓或桶。这样训练过程更快，并降低了内存开销。

4、分类提升算法（CatBoost）

顾名思义，CatBoost是一种处理数据中的分类变量的 Boosting 。大多数机器学习算法无法处理数据中的字符串或类别。因此，将分类变量转换为数值是一个重要的预处理步骤。

CatBoost可以在内部处理数据中的分类变量。使用有关特征组合的各种统计信息，将这些变量转换为数值变量。

如果你想了解如何将这些类别转换为数字，请阅读以下文章：
https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html#algorithm-main-stages_cat-to-numberic）

CatBoost被广泛使用的另一个原因是，它可以很好地处理默认的超参数集。因此，作为用户，我们不必花费大量时间来调整超参数。



在本文中，我们介绍了集成学习的基础知识，并研究了4种 Boosting 。有兴趣学习其他集成学习方法吗？你应该查看以下文章：

综合学习综合指南（附Python代码）：
https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/?utm_source=blog&utm_medium=4-boosting-algorithms-machine-learning
