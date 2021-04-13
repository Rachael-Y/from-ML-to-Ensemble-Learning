**简单集成技术**

- **最大投票法**
- **平均法**
- **加权平均法**

### **2.1 最大投票法**

最大投票方法通常用于分类问题。这种技术中使用多个模型来预测每个数据点。每个模型的预测都被视为一次“投票”。大多数模型得到的预测被用作最终预测结果。



例如，当你让5位同事评价你的电影时（最高5分）; 我们假设其中三位将它评为4，而另外两位给它一个5。由于多数人评分为4，所以最终评分为4。你可以将此视为采用了所有预测的众数（mode）。



最大投票的结果有点像这样：



![图片](https://mmbiz.qpic.cn/mmbiz_jpg/wc7YNPm3YxVnnFDjnsBK2g0IJkEqu3Jh3cgkOibVEfojV9PHHXe7cvwQqp2ffhJgAAcrCFjZXjC2dmdIIp1aC6g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



示例代码：



这里x_train由训练数据中的自变量组成，y_train是训练数据的目标变量。验证集是x_test（自变量）和y_test（目标变量）。



```
model1 = tree.DecisionTreeClassifier()

model2 = KNeighborsClassifier()

model3= LogisticRegression()

model1.fit(x_train,y_train)

model2.fit(x_train,y_train)

model3.fit(x_train,y_train)

pred1=model1.predict(x_test)

pred2=model2.predict(x_test)

pred3=model3.predict(x_test)

final_pred = np.array([])

for i in range(0,len(x_test)):

    final_pred =np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))
```



或者，你也可以在sklearn中使用“VotingClassifier”模块，如下所示：



```
from sklearn.ensemble import VotingClassifier

model1 = LogisticRegression(random_state=1)

model2 = tree.DecisionTreeClassifier(random_state=1)

model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')

model.fit(x_train,y_train)

model.score(x_test,y_test)
```



### **2.2 平均法**



类似于最大投票技术，这里对每个数据点的多次预测进行平均。在这种方法中，我们从所有模型中取平均值作为最终预测。平均法可用于在回归问题中进行预测或在计算分类问题的概率时使用。



例如，在下面的情况中，平均法将取所有值的平均值。



即（5 + 4 + 5 + 4 + 4）/ 5 = 4.4



![图片](https://mmbiz.qpic.cn/mmbiz_jpg/wc7YNPm3YxVnnFDjnsBK2g0IJkEqu3JhYheicwwyvdricnMiaP08E4ECibibiaCQVGibeQ1acI5yVdr11FxicFnagFjJibQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



示例代码：



```
model1 = tree.DecisionTreeClassifier()

model2 = KNeighborsClassifier()

model3= LogisticRegression()

model1.fit(x_train,y_train)

model2.fit(x_train,y_train)

model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)

pred2=model2.predict_proba(x_test)

pred3=model3.predict_proba(x_test)

finalpred=(pred1+pred2+pred3)/3
```



### **2.3 加权平均法**



这是平均法的扩展。为所有模型分配不同的权重，定义每个模型的预测重要性。例如，如果你的两个同事是评论员，而其他人在这方面没有任何经验，那么与其他人相比，这两个朋友的答案就更加重要。



计算结果为[（5 * 0.23）+（4 * 0.23）+（5 * 0.18）+（4 * 0.18）+（4 * 0.18）] = 4.41。



![图片](https://mmbiz.qpic.cn/mmbiz_png/wc7YNPm3YxVnnFDjnsBK2g0IJkEqu3JhiczUFNU6oweLgstElay81fvwzlDvEC6QybwTMc7fc6ibJFLBAubGglgw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

示例代码：



```
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)
```
