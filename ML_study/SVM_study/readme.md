### 学习内容
- SVM 硬间隔原理

- SVM 软间隔

- SMO 求解SVM

- 代码设计

  ##### 拓展阅读：

-  各种机器学习算法的应用场景分别是什么（比如朴素贝叶斯、决策树、K 近邻、SVM、逻辑回归最大熵模型）？ - xyzh的回答 - 知乎 https://www.zhihu.com/question/26726794/answer/151282052 

- https://zhuanlan.zhihu.com/p/25327755


### 1、硬间隔

本文是需要一定基础才可以看懂的，建议先看看参考博客，一些疑惑会在文中直接提出，大家有额外的疑惑可以直接评论，有问题请直接提出，相互交流。
### SVM-统计学习基础
一开始讲解了<code>最小间距超平面：所有样本到平面的距离最小</code>。而距离度量有了函数间隔和几何间隔，函数间隔与法向量$w$和$b$有关，$w$变为$2w$则函数间距变大了，于是提出了几何距离，就是对$w$处理，除以$||w||$，除以向量长度，从而让几何距离不受影响。

但是支持向量机提出了最大间隔分离超平面，这似乎与上面的分析相反，其实这个最大间隔是个什么概念呢？通过公式来分析一下，正常我们假设超平面公式是：
$$
w^{T}x+b=0  // 超平面
$$
$$
\max \limits_{w,b}   \quad  \gamma \\
s.t. \quad y_i(\frac{w}{||w||}x_i+\frac{b}{||w||}) > \gamma
$$
也就是说对于所有的样本到超平面距离 都大于$\gamma$，那这个$\gamma$如何求解，文中约定了概念支持向量：正负样本最近的两个点，这两个点之间的距离就是$\gamma$，那么问题来了，这中间的超平面有无数个，如何确定这个超平面呢？于是我们可以约束这个超平面到两个最近的点的距离是一样的。
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL2NvbW1vbi8lRTglQjYlODUlRTUlQjklQjMlRTklOUQlQTIucG5n?x-oss-process=image/format,png)
上图中两个红色菱形点与一个蓝色实心圆点就是支持向量，通过这个求解目标，以及约束条件来求解这个超平面。书中有完整的公式装换以及证明这个超平面的唯一性。

这里要讲解一个样本点到直线的距离，
正常我们可能难以理解公式里$y$去哪里了，拿二维空间做例子，正常我们说一个线性方程都是$y=ax+b$，其中a和b都是常量，这个线性方程中有两个变量$x$和$y$，转换公式就是$y-ax-b=0$，从线性矩阵的角度来思考问题就是 $y$是$x_1$，$x$是$x_2$，用一个$w^T$来表示这两者的系数，用$b$代替$-b$，所以公式就变为了：
$$
w^{T}x+b=0
$$
于是任意一个样本点到超平面的距离是：
$$
r = \frac{|w^{T}x+b|}{||w||}
$$
也就是说约束条件中要求$>\gamma$，其实就是大于支持向量到超平面的距离。

通过一个例子来看看：
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL2NvbW1vbi8lRTYlOTQlQUYlRTYlOEMlODElRTUlOTAlOTElRTklODclOEYlRTYlOUMlQkEtJUU0JUJFJThCJUU1JUFEJTkwJUU2JTg4JUFBJUU1JTlCJUJFLnBuZw?x-oss-process=image/format,png)
这里例子中有$w_1,w_2$，这是因为坐标点是二维的，相当于样本特征是两个，分类的结果是这两个特征的结果标签，所以这里的$w$就是一个二维的，说明在具体的应用里需要根据特征来确定$w$的维度。

##### 对偶讲解
其实原始问题是这样的：
$$
\max \limits_{w,b}   \quad  \gamma \\
s.t. \quad y_i(\frac{w}{||w||}x_i+\frac{b}{||w||}) > \gamma
$$
利用几何距离与函数距离的关系$\gamma = \frac{\hat{ \gamma}}{||w||}$将公式改为：
$$
\max \limits_{w,b}   \quad   \frac{\hat{ \gamma}}{||w||} \\
s.t. \quad y_i(wx_i+b) > \hat{\gamma}
$$
函数间隔是会随着$w与b$的变化而变化，同时将$w与b$变成$\lambda w与\lambda b$，则函数间隔也会变成$\lambda  \gamma$，所以书中直接将$\gamma=1$来转换问题。同样的问题又改为：
$$
\max \limits_{w,b}   \quad   \frac{1}{||w||} \\
s.t. \quad y_i(wx_i+b) >1
$$
求解最大值改为另一个问题，求解最小值：
$$
\min   \quad   \frac{1}{2} ||w||^2 \\
s.t. \quad y_i(wx_i+b) >1
$$
 支持向量满足： 

 ![img](https://img-blog.csdn.net/20180529201927600?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI2Nzk3MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

这就是一个对偶问题的例子，也是书中支持向量机模型的一个目标函数转换的过程，大家可以看看了解一下这个思路。其实书中利用拉格朗日乘子来求解条件极值，这一块在<code>高等数学中多元函数的极值及求解方法</code>中有提到。 
$$
\max   \quad   \frac {1}{||w||} \\
s.t. \quad y_i(w^{T}x_i+b) \ge {1} \\
i=1,...,n
$$
等价问题：
$$
\min \quad \frac{1}{2}||w||^{2} \\
s.t. \quad y_i(w^{T}x_i+b) \ge {1} \\
i=1,...,n
$$
这是一个凸优化问题：一个二次函数的优化问题，能直接用现成的优化计算包求解，但这里用一种更高效的方法：拉格朗日乘数法。
$$
\ L(w,b,a)= \frac {1}{2}||w||^{2} - \sum_{i=1}^{N}a_{i}(y_{i}(w^{T}x_{i}+b)-1)
$$
我们设：$ \theta(w) = \Max_{a_{i}}$
$$
\theta(w) = \ Max_{a_{i} \ge {0}} \ L(w,b,a)
$$
为什么要设$ \ Max L(w,b,a) $,这是因为拉格朗日的对偶性：

(1)$ x $满足条件：是支持向量，则
$$
y_i(w^{T}x_i+b) = {1} \\
L(w,b,a) = \frac {1}{2} ||w||^{2}
$$
(2)样本$ x $满足条件另一种形式$ y_i(w^{T}x_i+b)  >  {1}  $,则
$$
\ Max \quad L(w,b,a) = \frac {1}{2}||w||^{2} - \sum_{i=1}^{N}a_{i}c_{i} \\
另a_{i}=0,则 \quad Max \quad L(w,b,a) = \frac {1}{2}||w||^{2}
$$
(3)不满足约束条件：$ y_i(w^{T}x_i+b)  <  {1}  $

则：
$$
\ Max \quad L(w,b,a) = +\infty \quad 因为a_{i}可以取到无穷
$$
所以拉格朗日之后的问题变为：
$$
\min_{w,b}\max_{a_{i} \ge {0}}L(b,a,w)=p^{*}
$$
我们交换min和max的位置
$$
\max_{w,b}\min_{a_{i} \ge {0}}L(b,a,w)=d^{*} \\
虽然不是等价问题，但 \quad d^{*} \le p^{*}
$$
所以满足下限$ \d^{*} $ 的w,b也一定满足上限$ \p^{*} $

分别对w和b求偏导令其为0，将得到的关系式代入上面方程得：
$$
\quad  L(w,b,a) =  \sum_{i=1}^{N}a_i - \frac{1}{2}\sum_{i=1}^{N} \sum_{j=1}^{N}a_{i}a_{j}y_{i}y_{j}(x_i^{T} * x_j)  \\
s.t. \quad  \sum_{i}^{N}a_iy_i=0 \\
\quad  0\le a_i,i=1,...,N
$$

$$
则\quad  \max_{a_{i} \ge {0}} L(w,b,a) =  \sum_{i}a_i - \frac{1}{2}\sum_{i} \sum_{j}a_{i}a_{j}y_{i}y_{j}(x_i * x_j)  \\
求上式的极大值变为\quad \quad \min \frac{1}{2}\sum_{i} \sum_{j}a_{i}a_{j}y_{i}y_{j}(x_i * x_j) \\
s.t. \quad  \sum_{i}^{N}a_iy_i=0 \\
\quad  0\le a_i,i=1,...,N \\
$$

所以等价问题变为：
$$
\quad  \min_{a_{i} \ge {0}} L(w,b,a) =  \frac{1}{2}\sum_{i} \sum_{j}a_{i}a_{j}y_{i}y_{j}(x_i * x_j) - \sum_{i}a_i  \\
s.t. \quad  \sum_{i}^{N}a_iy_i=0 \\
\quad  0\le a_i,i=1,...,N \\
$$
上边求偏导得：
$$
w = \sum_{i=1}^{N}a_{i}y_{i}x_{i}
$$

$$
所以 \quad \quad f(x) = (\sum_{i=1}^{N}a_{i}y_{i}x_{i})^{T}x + b=\sum_{i=1}^{N}a_{i}y_{i}(x_{i}^{T}x)+b
$$

如果我们要预测新的样本点x，只需要计算x和xi的内积即可，这非常重要，是后面使用核函数进行非线性推广的前提。

这里说明一下，并不是所有的点都参与计算内积，因为满足约束条件的ai都为0， 只有支持向量上的点ai不为0需要参与内积计算。

 上述过程需满足KKT(Karush-Kuhn-Tucker)条件，即要求 

 ![img](https://img-blog.csdn.net/20180529202245998?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI2Nzk3MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

于是，对任意训练样本 ![img](https://img-blog.csdn.net/20180529202356270?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI2Nzk3MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) ，总有 ![img](https://img-blog.csdn.net/20180529202404694?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI2Nzk3MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

```python
这显示出支持向量机的一个重要性质：训练完成后，大部分的训练样本都不需保留，最终模型仅与支持向量有关。
```
### 软间隔

硬间隔是方便用来分隔线性可分的数据，如果样本中的数据是线性不可分的呢？也就是如图所示：
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL2NvbW1vbi8lRTglQkQlQUYlRTklOTclQjQlRTklOUElOTQtJUU1JTlCJUJFLnBuZw?x-oss-process=image/format,png)
有一部分红色点在绿色点那边，绿色点也有一部分在红色点那边，所以就不满足上述的约束条件：$s.t. \quad y_i(x_i+b) >1$，软间隔的最基本含义同硬间隔比较区别在于允许某些样本点不满足原约束，从直观上来说，也就是“包容”了那些不满足原约束的点。软间隔对约束条件进行改造，迫使某些不满足约束条件的点作为损失函数，如图所示：
![软间隔公式](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL2NvbW1vbi8lRTglQkQlQUYlRTklOTclQjQlRTklOUElOTQtJUU1JTg1JUFDJUU1JUJDJThGNy5QTkc?x-oss-process=image/format,png)
这里要区别非线性情况，例如：非线性就是一个圆圈，圆圈里是一个分类结果，圆圈外是一个分类结果。这就是非线性的情况。

其中当样本点不满足约束条件时，损失是有的，但是满足条件的样本都会被置为0，这是因为加入了转换函数，使得求解min的条件会专注在不符合条件的样本节点上。

但截图中的损失函数非凸、非连续，数学性质不好，不易直接求解，我们通常用一些其他函数来代替它，称为替代损失函数（surrogate loss）。后面采取了松弛变量的方式，来使得某些样本可以不满足约束条件。
![软间隔公式](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL2NvbW1vbi8lRTglQkQlQUYlRTklOTclQjQlRTklOUElOTQtJUU1JUI4JUI4JUU3JTk0JUE4JUU2JThEJTlGJUU1JUE0JUIxJUU1JTg3JUJEJUU2JTk1JUIwLSVFOCVCRiU4NyVFNiVCQiVBNCVFNCVCOCU4RCVFNyVBQyVBNiVFNSU5MCU4OCVFNyVCQSVBNiVFNiU5RCU5RiVFNiU5RCVBMSVFNCVCQiVCNiVFNyU5QSU4NCVFNyU4MiVCOS5QTkc?x-oss-process=image/format,png)
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL2NvbW1vbi8lRTUlQkMlOTUlRTUlODUlQTUlRTYlOUQlQkUlRTUlQkMlOUIlRTUlOEYlOTglRTklODclOEYlRTglQkQlQUYlRTklOTclQjQlRTklOUElOTQlRTUlODUlQUMlRTUlQkMlOEYuUE5H?x-oss-process=image/format,png)

这里思考一个问题：既然是线性不可分，难道最后求出来的支持向量就不是直线？某种意义上的直线？
其实还是直线，不满足条件的节点也被错误的分配了，只是尽可能的求解最大间隔，

### 核函数
引入核函数可以解决非线性的情况：<font color="#F00" size = "5px">将样本从原始空间映射到一个更高为的特征空间，使得样本在这个特征空间内线性可分</font>。图片所示：

 ![SVM-非线性样本可分图.PNG](https://img-blog.csdn.net/20180529202428223?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI2Nzk3MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

粉红色平面就是超平面，椭圆形空间曲面就是映射到高维空间后的样本分布情况，为了将样本转换空间或者映射到高维空间，我们可以引用一个映射函数，将样本点映射后再得到超平面。这个技巧不仅用在SVM中，也可以用到其他统计任务。
但映射函数并不是最重要的，核函数是重要的，看到《统计学习方法》中提到的概念：
![核函数定义](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL2NvbW1vbi8lRTYlQTAlQjglRTUlODclQkQlRTYlOTUlQjAlRTUlQUUlOUElRTQlQjklODkucG5n?x-oss-process=image/format,png)

 ![img](https://img-blog.csdn.net/20180529202559652?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI2Nzk3MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

其中映射函数与核函数之间有函数关系，一般我们显式的定义核函数，而不显式的定义映射函数，一方面是因为计算核函数比映射函数简单。我们对一个二维空间做映射，选择的新空间是原始空间的所有一阶和二阶的组合，得到了五个维度；如果原始空间是三维，那么我们会得到 19 维的新空间，这个数目是呈爆炸性增长的，这给 的计算带来了非常大的困难，而且如果遇到无穷维的情况，就根本无从计算了。所以就需要 Kernel 出马了。这样，<font color="#F00" size = "5px">核函数：一个**对称函数**所对应的**核矩阵半正定，**它就可作为核函数使用。

  事实上，对于一个半正定核矩阵，总能找到一个与之对应的映射Ф。换言之，**任何一个核函数**都隐式地定义了一个称为"**再生核希尔伯特空间**"的特征空间。反之同样确定了一个特征空间，其映射函数也可能是不一样的</font>。举个例子：
![核函数与映射函数](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL2NvbW1vbi8lRTYlQTAlQjglRTUlODclQkQlRTYlOTUlQjAlRTQlQjglOEUlRTYlOTglQTAlRTUlQjAlODQlRTUlODclQkQlRTYlOTUlQjAlRTQlQjklOEIlRTklOTclQjQucG5n?x-oss-process=image/format,png)
上述例子很好说明了核函数和映射函数之间的关系。这就是核技巧，将原本需要确定映射函数的问题转换为了另一个问题，从而减少了计算量，也达到了线性可分的目的，

```
原始方法：  样本X   ---->  特征空间Z  ---- >   内积求解超平面
核函数：    样本X   ---- >   核函数 求解超平面
```
但是我一直很疑惑，为什么这个核函数就正好是映射函数的内积？他为什么就可以生效？核函数的参数是哪里来的？就是样本中的两个样本点 ?

 ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy84NTcwNzA0LTYzMjczODdkYmE0YWUwYmEucG5nP2ltYWdlTW9ncjIvYXV0by1vcmllbnQvc3RyaXB8aW1hZ2VWaWV3Mi8yL3cvNjMyL2Zvcm1hdC93ZWJw?x-oss-process=image/format,png) 

 从上例可以看出，核函数一定，映射函数是不唯一的，而且当维度是无线大的时候，我们几乎无法求得映射函数，那么核函数的作用就在于此，核函数避免了映射函数的求解，叫做核技巧。

核函数的特征： 

 ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy84NTcwNzA0LTA1YTE3ZGQyMzY5N2I5YmYucG5nP2ltYWdlTW9ncjIvYXV0by1vcmllbnQvc3RyaXB8aW1hZ2VWaWV3Mi8yL3cvNjUyL2Zvcm1hdC93ZWJw?x-oss-process=image/format,png) 

  对于原空间中的非线性可分问题，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分。若将原始的二维空间映射到一个合适的三维空间 ，就能找到一个合适的划分超平面。幸运的是，如果原始空间是有限维，即属性数有限，那么一定存在一个高维特征空间使样本可分。

 ![img](https://img-blog.csdn.net/20180529202518475?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI2Nzk3MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

 ![img](https://img-blog.csdn.net/20180529202535270?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI2Nzk3MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 
这也说明核函数是作用在两个样本点上的，以下来自如某一篇博客里：

> **最理想的情况下，我们希望知道数据的具体形状和分布，从而得到一个刚好可以将数据映射成线性可分的 ϕ(⋅) ，然后通过这个 ϕ(⋅) 得出对应的 κ(⋅,⋅) 进行内积计算。然而，第二步通常是非常困难甚至完全没法做的。不过，由于第一步也是几乎无法做到，因为对于任意的数据分析其形状找到合适的映射本身就不是什么容易的事情，所以，人们通常都是“胡乱”选择映射的，所以，根本没有必要精确地找出对应于映射的那个核函数，而只需要“胡乱”选择一个核函数即可——我们知道它对应了某个映射，虽然我们不知道这个映射具体是什么。由于我们的计算只需要核函数即可，所以我们也并不关心也没有必要求出所对应的映射的具体形式。**

#### 常用的核函数及对比：
- Linear Kernel 线性核
   $$
   k(x_i,x_j)=x_i^{T}x_j
   $$
  线性核函数是最简单的核函数，主要用于线性可分，它在原始空间中寻找最优线性分类器，具有参数少速度快的优势。 如果我们将线性核函数应用在KPCA中，我们会发现，推导之后和原始PCA算法一模一样，这只是线性核函数偶尔会出现等价的形式罢了。
  
- Polynomial Kernel 多项式核
  $$
   k(x_i,y_j)=(x_i^{T}x_j)^d \\
  也有复杂的形式：\\
 k(x_i,x_j)=(ax_i^{T}x_j+b)^d
  $$
  
  其中$d\ge1$为多项式次数，参数就变多了，多项式核实一种非标准核函数，它非常适合于正交归一化后的数据，多项式核函数属于全局核函数，可以实现低维的输入空间映射到高维的特征空间。其中参数d越大，映射的维度越高，和矩阵的元素值越大。故易出现过拟合现象。

- 径向基函数 高斯核函数 Radial Basis Function（RBF）

$$
   k(x_i,x_j)=exp(-\frac{||x_i-x_j||^2}{2\sigma^2})
$$

$\sigma>0$是高斯核带宽，这是一种经典的鲁棒径向基核，即高斯核函数，鲁棒径向基核对于数据中的噪音有着较好的抗干扰能力，其参数决定了函数作用范围，超过了这个范围，数据的作用就“基本消失”。高斯核函数是这一族核函数的优秀代表，也是必须尝试的核函数。对于大样本和小样本都具有比较好的性能，因此在多数情况下不知道使用什么核函数，优先选择径向基核函数。

- Laplacian Kernel 拉普拉斯核
$$
   k(x_i,x_j)=exp(-\frac{||x_i-x_j||}{\sigma})
$$

- Sigmoid Kernel Sigmoid核
$$
   k(x_i,x_j)=tanh(\alpha x^Tx_j+c)
$$
采用Sigmoid核函数，支持向量机实现的就是一种多层感知器神经网络。


其实还有很多核函数，在参考博客里大家都可以看到这些核函数，对于核函数如何选择的问题，吴恩达教授是这么说的：
- 如果Feature的数量很大，跟样本数量差不多，这时候选用LR或者是Linear Kernel的SVM

- 如果Feature的数量比较小，样本数量一般，不算大也不算小，选用SVM+Gaussian Kernel

- 如果Feature的数量比较小，而样本数量很多，需要手工添加一些feature变成第一种情况

   ![img](https://img-blog.csdn.net/20180529202623398?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI2Nzk3MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 






### 2、软间隔


###  前言
之前写的一偏文章主要是[SVM的硬间隔](https://blog.csdn.net/randompeople/article/details/90020648)，结合[SVM拉格朗日对偶问题](https://blog.csdn.net/randompeople/article/details/92083294)可以求解得到空间最大超平面，但是如果样本中与较多的异常点，可能对样本较敏感，不利于模型泛化，于是有了软间隔的支持向量机形式，本文来了解一下此问题。

### 软间隔最大化
引入松弛变量，使得一部分异常数据也可以满足约束条件：$y_i(x_i+b) >=1 - \varepsilon_i$，既然约束条件引入了松弛变量，那么点到超平面的距离是不是也要改变，于是调整为：
$$
\min   \quad   \frac{1}{2} ||w||^2+C\sum_{i}^{N}\varepsilon_i \\
s.t. \quad y_i(x_i+b) \ge 1 - \varepsilon_i \qquad  \text{i=1,2...,n}\\
\varepsilon_i  \ge 0
$$
- C：表示惩罚因子，这个值大小表示对误分类数据集的惩罚，调和最大间距和误分类点个数之间的关系。
- $\varepsilon_i$：也作为代价。

这也是一个凸二次规划问题，可以求解得到$w$，但b的求解是一个区间范围，让我们来看看是怎么回事，求解流程跟硬间隔没差别，直接得到拉格朗日对偶问题：


$$
\max_{a_i>0,\mu>0} \min_{w_i,b,\varepsilon}   \quad   L(w,b,\varepsilon,a,\mu)= \frac{1}{2} ||w||^2+C\sum_{i}^{N}\varepsilon_i+\sum_{i=1}^{N}a_{i}[1-y_i(wx_i+b)+\varepsilon_i]+\sum_{i}^{N} \mu_i \varepsilon_i
$$
继续按照流程走：
- 对w、b、$\varepsilon$ 求偏导，让偏导等于0，结果为：
$$
w = \sum_{i}a_iy_ix_i \\
\sum_{i}a_iy_i = 0 \\
C-a_i-u_i =0
$$
- 代入上面的方程得到：

$$
\max_{a_i>0,\mu>0} \quad  L(w,b,\varepsilon,a,\mu) =  -\frac{1}{2}\sum_{i} \sum_{j}a_{i}a_{j}y_{i}y_{j}(x_i * x_j) + \sum_{i}a_i \\
s.t. \quad  \sum_{i}^{N}a_iy_i=0 \\
\quad  0\le a_i\le C
$$
去掉符号，将max 转换为 min ：
$$
\min_{a_i>0,\mu>0} \quad  L(w,b,\varepsilon,a,\mu) =  \frac{1}{2}\sum_{i} \sum_{j}a_{i}a_{j}y_{i}y_{j}(x_i * x_j) - \sum_{i}a_i \\
s.t. \quad  \sum_{i}^{N}a_iy_i=0 \\
\quad  0\le a_i\le C
$$
这里代入之后就只有一个因子$a_i$，对此方程求解$a_i$
- w、b:
$$
w = \sum_{i}a_iy_ix_i \\
$$
b的计算就需要思考了，选取满足$\quad  0\le a_i\le C$的$a_i$，利用这些点来求解b：
$$
b = y_j-\sum_{i}a_iy_i(x_i*x_j)
$$
当然符合这个条件的也不只有一个，存在多个条件。求解平均值作为一个唯一值。

- 超平面
$$
y = wx+b
$$

和上一篇的硬间隔最大化的线性可分SVM相比，多了一个约束条件：$0\le a_i \le C$。
$$
0\le a_i\le C
$$



### 3、SMO求解SVM

SVM算法中目标函数最终是一个关于*a* 向量的函数。 

### SMO算法

- ### 前言
   SVM算法中目标函数最终是一个关于$a$向量的函数。本文将通过SMO算法来极小化这个函数。
   
   ### SMO算法
   首先我们再写一下带核函数的优化目标：
   $$
   \underbrace{min}_{a} \frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m a_ia_jy_iy_jK(x_i,x_j)-\sum_{i=1}^ma_i\\
   s.t \sum_{i=1}^ma_i=0\\
   0\le a_i \le C
   $$
   SMO算法则采用了一种启发式的方法。它每次只优化两个变量，将其他的变量都视为常数。由于$\sum_{i=1}^ma_i=0$.假如将$a_3,a_4,...,a_m$固定，那么$a_1$，$a_2$之间的关系也确定了。这样SMO算法将一个复杂的优化算法转化为一个比较简单的两变量优化问题。为了后面表示方便，定义：$K_{ij}=\phi(x_i)  
   \cdot \phi(x_j)$
   
   所以假设我们选择了两个变量：$a_1,a_2$来作为第一个选择的变量来迭代更新，当然怎么选择也有有原因的，这个后面会讲解。于是其他变量都是常量，那么公式就变为：
   $$
   \underbrace{min}_{a_i,a_j}\qquad a_1a_2y_1y_2K_{12}+\frac{1}{2}a_1^2K_{11}+\frac{1}{2}a_2^2K_{22}+\frac{1}{2}a_1y_1\sum_{j=3}^m a_jy_jK_{1,j}+\frac{1}{2}a_2y_2\sum_{j=3}^m a_jy_jK_{2,j}-a_1-a_2-\sum_{j=3}^ma_j\\
   s.t \qquad a_1y_1+a_2y_2=-\sum_{i=3}^ma_i=\varepsilon \\
   0\le a_i \le C，i=1,2
   $$
   因为$a_1y_1+a_2y_2=\varepsilon$，所以得到：$a_1=\varepsilon y_1-a_2y_2y_1$，把$a_1$代入上式，并且为来简单描述，我们必须还用到一些变量来替换公式中的某一类变量：
   $$
   g(x)=w\phi(x_i) +b=\sum_{j=1}^ma_jy_jK(x,x_j)+b \\
   v_i=\sum_{j=3}^m a_jy_jK_{ij}=g(x)-\sum_{j=1}^2 a_jy_jK_{ij}-b=g(x)-a_1y_1K_{i1}-a_2y_2K_{i2}-b
   $$
   
   这样$a_1$、$v_1$、$v_2$代入上式，得到$W(a_1,a_2)$：
   $$
   W(a_1,a_2)=(\varepsilon y_1-a_2y_2y_1)a_2y_1y_2K_{12}+\frac{1}{2}(\varepsilon y_1-a_2y_2y_1)^2K_{11}+\frac{1}{2}a_2^2K_{22}+\frac{1}{2}(\varepsilon y_1-a_2y_2y_1)y_1v_1+\frac{1}{2}a_2y_2v_2-(\varepsilon y_1-a_2y_2y_1)-a_2-\sum_{j=3}^ma_j
   $$
   现在$W(a_1,a_2)$公式中只有一个变量$a_2$，求最小值，于是，我们求导：
   $$
   \frac{\partial W}{\partial a_2}=K_{11}a_2+K_{12}a_2-2K_{12}a_2-K_{11}\varepsilon y_2+K_{12}\varepsilon y_2+y_1 y_2 -1- v_1y_2+y_2v_2
   $$
   让$\frac{\partial W}{\partial a_2}=0$，得到：
   $$
   a_2=\frac{y_2(y_2-y_1+K_{11}\varepsilon -K_{12}\varepsilon+v_1-v_2)}{K_{11}+K_{22}+2K_{12}}
   $$
   
   这样似乎也算是完事了，但变换一下似乎更好一些，于是，提出一个假设变量$E_i$：
   $$
   E_i=g(x_i)-y_i=\sum_{j=1}^ma_jy_jK(x_i,x_j)+b-y_i
   $$
   
   将变量$E_i$以及$a_1y_1+a_2y_2=\varepsilon$代入公式中，有了如下的变换结果：
   $$
   (K_{11}+K_{22}+2K_{12})a_2=(K_{11}+K_{22}+2K_{12})a_2^{old}+y_2(E_1-E_2)
   $$
   于是$a_2^{new,unc}$：
   $$
   a_2^{new,unc}=a_2^{old}+\frac{y_2(E_1-E_2)}{K_{11}+K_{22}+2K_{12}}
   $$
   
   
   
   ### $a_2^{new,unc}$及L和H的定义
   
    $a_2^{new,unc}$的值并不是最终的更新函数值，因为由于$a_1$、$a_2$的关系被限制在盒子里的一条线段上，所以两变量的优化问题实际上仅仅是一个变量的优化问题。不妨我们假设最终是$a_2$的优化问题。由于我们采用的是启发式的迭代法，假设我们上一轮迭代得到的解是$a_1^{old}$、$a_2^{old}$，假设沿着约束方向$a_2$未经剪辑的解是$a_2^{new,unc}$，本轮迭代完成后的解为$a_1^{new}$、$a_2^{new}$，$a_2^{new}$必须满足上图中的线段约束：
   ![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL1JOTi9sZWZ0LnBuZw?x-oss-process=image/format,png)
   
   
   首先 L 的定义是$a_2$的下限：图中红线处(0,-k)点出表示$a_2$的红线处的最小值，于是$L=-k$，因为$a_1-a_2=k$，$-k=a_2-a_1$，所以$L=-k=a_2-a_1$，
   同理对于H，图中黑线(C,C-k)，所以$H=C-k$，由于$a_1-a_2=k$，所以$H=C-k=C-a_1+a_2$
   同时是由于在迭代过程中更新，所以换成上一轮的值：$L=a_2^{old}-a_1^{old}$、$H=C-k=C-a_1^{old}+a_2^{old}$。综合这两种情况，最终得到：
   
   $$
   L=max(0,a_2^{old}-a_1^{old}) \\
   H=min(C,C-a_1^{old}+a_2^{old})
   $$
   ![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL1JOTi9yaWdodC5wbmc?x-oss-process=image/format,png)
   L相关的值在红色线上的点(C,k-C)，和黑色线的点(k,0)，所以：
   $$
   L=max(0,a_1^{old}+a_2^{old}-C)
   $$
   H相关的值在红色线的点(k-C,C)和黑色线的点(0,k)，所以：
   $$
   H=min(C,a_1^{old}+a_2^{old})
   $$
   
   
   这其实就是限制了$a_i$的范围，所以选取了两个变量更新之后，某个变量的值更新都满足这些条件，假设求导得到$a_2^{new,unc}$，于是得到了：
   $$
   a_2^{new} =
   \begin{cases}
   H& a_2^{new,unc}>H\\
   a_2^{new,unc}& L<a_2^{new,unc}<H\\
   L& a_2^{new,unc}<L
   \end{cases}
   $$
   其中
   
   ### 更新b向量
   得到$a_1^{new}$和$a_2^{new}$之后，我们可以来计算$b$，我们先用$a_1^{new}$来得到最新的$b^{new}$，当$0<a_i^{new} \le C$时，
   $$
   y_1-\sum_{i=1}^{m}a_iy_iK_{i1}-b_1=0\\
   $$
   转换后得到：
   $$
   b_1=y_1-\sum_{i=3}^{m}a_iy_iK_{i1}-a_1^{new}y_1K_{11}-a_2^{new}y_2K_{21}
   $$
   同时由于$E_1=g(x_1)-y_1=\sum_{j=1}^ma_jy_jK(x_i,x_j)+b-y_i=\sum_{j=3}^{m}a_jy_jK_{j1}+a_1^{old}y_1K_{11}+a_2^{old}y_2K_{21}+b^{old}-y_1$，替代上述得到更新后的$b_1^{new}$：
   $$
   b_1^{new}=-E_1-y_1K_{11}(a_1^{new}-a_1^{old})-y_2K_{21}(a_2^{new}-a_2^{old})+b^{old}
   $$
   同理可以得到$b_2^{new}$：
   $$
   b_2^{new}=-E_2-y_1K_{12}(a_1^{new}-a_1^{old})-y_2K_{22}(a_2^{new}-a_2^{old})+b^{old}
   $$
   那么问题来了，有两个$b$，这怎么弄？
   
   $$
   b^{new} =
   \begin{cases}
   \frac{b_1^{new}+b_2^{new}}{2}& a_2对应的点是支持向量 \\
   b_1^{new}& a_2对应的点不是支持向量
   \end{cases}
   $$
   为什么这么选择？
   ![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL1JOTi9iX25ldy5qcGc?x-oss-process=image/format,png)
   具体情况参考李航书130页，选择$b^{new}=(b_1+b_2)/2$这只是一个工程上的选择而已，因为没有一个确定的b，得到了$b^{new}$之后利用其来更新$E_i$，公式上述其实有提供，这里再明确一下：
   $$
   E_i^{new}=\sum_{S}y_ja_jK(x_i,x_j)+b^{new}-y_i
   $$
   这里的$S$是所有的支持向量$x_i$的集合，这里每一个样本$x_i$都对应一个$E_i$，利用$b^{new}$和$E_i^{new}$迭代更新。这里的E跟更新$a_i$的表达式场景不同，E1是用于某次迭代中，SMO选中的$a$对应的计算过程的，而$E_i$是迭代完毕后，尝试优化更新所有的$E_i$值的时候。如果只是计算，都是用的这个公式。为什么不用所有支持向量，因为不是支持向量的样本，其𝛼值为0，就算放在式子里面也是0，所以可以去掉。
   
   - 为什么求解$b^{new}$用$E^1$表示？
   用E1表示，是为了求出最终的bnew, 并求出Ei。
   ### 如何选择两个变量
   - 1、外层变量
   　　SMO算法称选择第一个变量为外层循环，这个变量需要选择在训练集中违反KKT条件最严重的样本点。对于每个样本点，解要满足的KKT条件的对偶互补条件：$a_i(y_i(w^Tx_i+b)-1+\xi_{i})=0$
   　　KKT条件我们在第一节已经讲到了： 
   $$
   a_i=0 \Rightarrow y_ig(x_i)\ge0 \\
   0<a_i<C \Rightarrow y_ig(x_i)=1 \\
   a_i=C  \Rightarrow y_ig(x_i)\le1
   $$
   这三个KKT条件，那么一般我们取违反$0<a_i<C \Rightarrow y_ig(x_i)=1$最严重的点，如果没有，再选择其他$a_i=0$、$a_i=C$的条件。
   - 2、内层变量
   SMO算法称选择第二一个变量为内层循环，假设我们在外层循环已经找到了$a_1$， 第二个变量$a_2$的选择标准是让$|E_1−E_2|$有足够大的变化。由于$a_1$定了的时候，$E_1$也确定了，所以要想$|E_1−E_2|$最大，只需要在$E_1$为正时，选择最小的$E_i$作为$E_2$， 在$E_1$为负时，选择最大的$E_i$作为$E_2$，可以将所有的$E_i$保存下来加快迭代。如果内存循环找到的点不能让目标函数有足够的下降， 可以采用以下步骤：
   （1）遍历支持向量点来做 $a_2$ , 直到目标函数有足够的下降；
   （2）如果所有的支持向量做 $a_2$ 都不能让目标函数有足够的下降，可以在整个样本集上选择 $a_2$；
   （3）如果整个样本集依然不存在，则跳回外层循环重新选择 $a_1$。
   
   这其中的$E_i$也在上面的更新$a_i$、$b$的代码里有提到，这里不详细介绍，大家参考博客来详细了解。
   
   
   ### 问题
   - 为什么每次更新Ei时只用支持向量的样本？在上面优化的时候例如E1使用的是全部的样本。Ei= g(xi)-yi=wx+b-y不是这个公式吗？
   这里两个E的表达式场景不同，E1是用于某次迭代中，SMO选中的$a$对应的计算过程的，而$E_i$是迭代完毕后，尝试优化更新所有的$E_i$值的时候。如果只是计算，都是你说的那个公式。回顾一下，w与αi,xi,yi的关系式为:
   
   w = ∑ αi*yi*xi ，其中i = 1,2,3,...,N
   我们初始化的α是一个全为0的向量，即α1=α2=α3=...=αN=0，w的值即为0.我们进行SMO算法时，每轮挑选出两个变量αi，固定其他的α值，也就是说，那些从来没有被挑选出来过的α，值始终为0，而根据前面所学，支持向量对应的αi是一定满足 0<αi<=C的.有了这个认识之后，为什么不用全集呢，因为不是支持向量的样本点，对应的αi值为0啊，加起来也没有意义，对w产生不了影响，只有支持向量对应的点 (xi,yi)与对应的αi相乘，产生的值才对w有影响啊。从这里也能理解，为什么李航书中，认为支持向量不仅仅是处于间隔边界上的点，还包括那些处于间隔边界和分类超平面之间、分类超平面之上、分类超平面错误的一侧处的点了，因为后面所说的那些点，对应的αi为C，对w的计算可以起到影响作用，换句话说，就是能对w起到影响作用的点，都属于支持向量!
   
   - 关于惩罚项C
   C是一个超参数，需要调参的。后面我有一篇文章专门讲到了C这个参数的调参。支持向量机高斯核调参小结： https://www.cnblogs.com/pinard/p/6126077.html选择两个变量的流程参见我4.1节和4.2节。最开始所有样本的alpha都是初始化为0的，后面按4.1节和4.2节的流程每轮选择2个样本的alpha进行迭代。
   
   
   ### 参考博客
   [支持向量机原理(四)SMO算法原理](https://www.cnblogs.com/pinard/p/6111471.html#!comments)
   [知乎](https://zhuanlan.zhihu.com/p/36535299)
   [线性支持向量机中KKT条件的讨论](https://blog.csdn.net/james_616/article/details/72869015)

### 4、代码实现

1.txt文件

```
1 0 1
-1 0 -1
0.971354 0.209317 1
-0.971354 -0.209317 -1
0.906112 0.406602 1
-0.906112 -0.406602 -1
0.807485 0.584507 1
-0.807485 -0.584507 -1
0.679909 0.736572 1
-0.679909 -0.736572 -1
0.528858 0.857455 1
-0.528858 -0.857455 -1
0.360603 0.943128 1
-0.360603 -0.943128 -1
0.181957 0.991002 1
-0.181957 -0.991002 -1
-3.07692e-06 1 1
3.07692e-06 -1 -1
-0.178211 0.970568 1
0.178211 -0.970568 -1
-0.345891 0.90463 1
0.345891 -0.90463 -1
-0.496812 0.805483 1
0.496812 -0.805483 -1
-0.625522 0.67764 1
0.625522 -0.67764 -1
-0.727538 0.52663 1
0.727538 -0.52663 -1
-0.799514 0.35876 1
0.799514 -0.35876 -1
-0.839328 0.180858 1
0.839328 -0.180858 -1
-0.846154 -6.66667e-06 1
0.846154 6.66667e-06 -1
-0.820463 -0.176808 1
0.820463 0.176808 -1
-0.763975 -0.342827 1
0.763975 0.342827 -1
-0.679563 -0.491918 1
0.679563 0.491918 -1
-0.57112 -0.618723 1
0.57112 0.618723 -1
-0.443382 -0.71888 1
0.443382 0.71888 -1
-0.301723 -0.78915 1
0.301723 0.78915 -1
-0.151937 -0.82754 1
0.151937 0.82754 -1
9.23077e-06 -0.833333 1
-9.23077e-06 0.833333 -1
0.148202 -0.807103 1
-0.148202 0.807103 -1
0.287022 -0.750648 1
-0.287022 0.750648 -1
0.411343 -0.666902 1
-0.411343 0.666902 -1
0.516738 -0.559785 1
-0.516738 0.559785 -1
0.599623 -0.43403 1
-0.599623 0.43403 -1
0.65738 -0.294975 1
-0.65738 0.294975 -1
0.688438 -0.14834 1
-0.688438 0.14834 -1
0.692308 1.16667e-05 1
-0.692308 -1.16667e-05 -1
0.669572 0.144297 1
-0.669572 -0.144297 -1
0.621838 0.27905 1
-0.621838 -0.27905 -1
0.551642 0.399325 1
-0.551642 -0.399325 -1
0.462331 0.500875 1
-0.462331 -0.500875 -1
0.357906 0.580303 1
-0.357906 -0.580303 -1
0.242846 0.635172 1
-0.242846 -0.635172 -1
0.12192 0.664075 1
-0.12192 -0.664075 -1
-1.07692e-05 0.666667 1
1.07692e-05 -0.666667 -1
-0.118191 0.643638 1
0.118191 -0.643638 -1
-0.228149 0.596667 1
0.228149 -0.596667 -1
-0.325872 0.528323 1
0.325872 -0.528323 -1
-0.407954 0.441933 1
0.407954 -0.441933 -1
-0.471706 0.341433 1
0.471706 -0.341433 -1
-0.515245 0.231193 1
0.515245 -0.231193 -1
-0.537548 0.115822 1
0.537548 -0.115822 -1
-0.538462 -1.33333e-05 1
0.538462 1.33333e-05 -1
-0.518682 -0.111783 1
0.518682 0.111783 -1
-0.479702 -0.215272 1
0.479702 0.215272 -1
-0.423723 -0.306732 1
0.423723 0.306732 -1
-0.353545 -0.383025 1
0.353545 0.383025 -1
-0.272434 -0.441725 1
0.272434 0.441725 -1
-0.183971 -0.481192 1
0.183971 0.481192 -1
-0.0919062 -0.500612 1
0.0919062 0.500612 -1
1.23077e-05 -0.5 1
-1.23077e-05 0.5 -1
0.0881769 -0.480173 1
-0.0881769 0.480173 -1
0.169275 -0.442687 1
-0.169275 0.442687 -1
0.2404 -0.389745 1
-0.2404 0.389745 -1
0.299169 -0.324082 1
-0.299169 0.324082 -1
0.343788 -0.248838 1
-0.343788 0.248838 -1
0.373109 -0.167412 1
-0.373109 0.167412 -1
0.386658 -0.0833083 1
-0.386658 0.0833083 -1
0.384615 1.16667e-05 1
-0.384615 -1.16667e-05 -1
0.367792 0.0792667 1
-0.367792 -0.0792667 -1
0.337568 0.15149 1
-0.337568 -0.15149 -1
0.295805 0.214137 1
-0.295805 -0.214137 -1
0.24476 0.265173 1
-0.24476 -0.265173 -1
0.186962 0.303147 1
-0.186962 -0.303147 -1
0.125098 0.327212 1
-0.125098 -0.327212 -1
0.0618938 0.337147 1
-0.0618938 -0.337147 -1
-1.07692e-05 0.333333 1
1.07692e-05 -0.333333 -1
-0.0581615 0.31671 1
0.0581615 -0.31671 -1
-0.110398 0.288708 1
0.110398 -0.288708 -1
-0.154926 0.251167 1
0.154926 -0.251167 -1
-0.190382 0.206232 1
0.190382 -0.206232 -1
-0.215868 0.156247 1
0.215868 -0.156247 -1
-0.230974 0.103635 1
0.230974 -0.103635 -1
-0.235768 0.050795 1
0.235768 -0.050795 -1
-0.230769 -1e-05 1
0.230769 1e-05 -1
-0.216903 -0.0467483 1
0.216903 0.0467483 -1
-0.195432 -0.0877067 1
0.195432 0.0877067 -1
-0.167889 -0.121538 1
0.167889 0.121538 -1
-0.135977 -0.14732 1
0.135977 0.14732 -1
-0.101492 -0.164567 1
0.101492 0.164567 -1
-0.0662277 -0.17323 1
0.0662277 0.17323 -1
-0.0318831 -0.173682 1
0.0318831 0.173682 -1
6.15385e-06 -0.166667 1
-6.15385e-06 0.166667 -1
0.0281431 -0.153247 1
-0.0281431 0.153247 -1
0.05152 -0.13473 1
-0.05152 0.13473 -1
0.0694508 -0.112592 1
-0.0694508 0.112592 -1
0.0815923 -0.088385 1
-0.0815923 0.088385 -1
0.0879462 -0.063655 1
-0.0879462 0.063655 -1
0.0888369 -0.0398583 1
-0.0888369 0.0398583 -1
0.0848769 -0.018285 1
-0.0848769 0.018285 -1
0.0769231 3.33333e-06 1
-0.0769231 -3.33333e-06 -1
```

```
import numpy as np

def clip(value, lower, upper):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value

def default_ker(x, z):
    return x.dot(z.T)

def svm_smo(x, y, ker, C, max_iter, epsilon=1e-5):
    # initialization
    n, _ = x.shape
    alpha = np.zeros((n,))
        
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = ker(x[i], x[j])
    
    iter = 0
    while iter <= max_iter:
        
        for i in range(n):
            # randomly choose an index j, where j is not equal to i
            j = np.random.randint(low=0, high=n-1)
            while (i==j): j = np.random.randint(low=0, high=n-1)
            
            # update alpha_i
            eta = K[j, j] + K[i, i] - 2.0 * K[i, j]
            if np.abs(eta) < epsilon: continue # avoid numerical problem
            
            e_i = (K[:, i] * alpha * y).sum() - y[i]
            e_j = (K[:, j] * alpha * y).sum() - y[j]
            alpha_i = alpha[i] - y[i] * (e_i - e_j) / eta
            
            # clip alpha_i
            lower, upper = 0, C
            zeta = alpha[i] * y[i] + alpha[j] * y[j]
            if y[i] == y[j]:
                lower = max(lower, zeta / y[j] - C)
                upper = min(upper, zeta / y[j])
            else:
                lower = max(lower, -zeta / y[j])
                upper = min(upper, C - zeta / y[j])
                
            alpha_i = clip(alpha_i, lower, upper)
            alpha_j = (zeta - y[i] * alpha_i) / y[j]
            
            alpha[i], alpha[j] = alpha_i, alpha_j
        
        iter += 1
    
    # calculate b
    b = 0
    for i in range(n):
        if epsilon < alpha[i] < C - epsilon:
            b = y[i] - (y * alpha * K[:, i]).sum()
    
    def f(X): # predict the point X based on alpha and b
        results = []
        for k in range(X.shape[0]):
            result = b
            for i in range(n):
                result += y[i] * alpha[i] * ker(x[i], X[k])
            results.append(result)
        return np.array(results)
    
    return f, alpha, b
```

```
def data_visualization(x, y):
    import matplotlib.pyplot as plt
    category = {'+1': [], '-1': []}
    for point, label in zip(x, y):
        if label == 1.0: category['+1'].append(point)
        else: category['-1'].append(point)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for label, pts in category.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)
    plt.show() 
```

```
import numpy as np
%matplotlib inline
# random a dataset on 2D plane
def simple_synthetic_data(n, n0=5, n1=5): # n: number of points, n0 & n1: number of points on boundary
    # random a line on the plane
    w = np.random.rand(2) 
    w = w / np.sqrt(w.dot(w))
    
    # random n points 
    x = np.random.rand(n, 2) * 2 - 1
    d = (np.random.rand(n) + 1) * np.random.choice([-1,1],n,replace=True) # random distance from point to the decision line, d in [-2,-1] or [1,2]. d=-1 or d=1 indicate the boundary in svm
    d[:n0] = -1
    d[n0:n0+n1] = 1
    
    # shift x[i] to make the distance between x[i] and the decision become d[i]
    x = x - x.dot(w).reshape(-1,1) * w.reshape(1,2) + d.reshape(-1,1) * w.reshape(1,2)
    
    # create labels
    y = np.zeros(n)
    y[d < 0] = -1
    y[d >= 0] = 1
    return x, y

x, y = simple_synthetic_data(200)
data_visualization(x, y)
```

 ![img](data:image/png;base64,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) 

```
def spiral_data():
    data = np.loadtxt('C:/Users/dell/Desktop/1.txt')
    x = data[:,:2]
    y = data[:,2]
    return x, y

x, y = spiral_data()
data_visualization(x, y)
```

 ![img](data:image/png;base64,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) 

```
# load the synthetic data
x, y = simple_synthetic_data(100, n0=5, n1=5)

# run svm classifier
ker = default_ker
model, alpha, bias = svm_smo(x, y, ker, 1e10, 1000)

# visualize the result
import matplotlib.pyplot as plt
category = {'+1': [], '-1': []}
for point, label in zip(x, y):
    if label == 1.0: category['+1'].append(point)
    else: category['-1'].append(point)
fig = plt.figure()
ax = fig.add_subplot(111)

# plot points
for label, pts in category.items():
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], label=label)

# calculate weight
weight = 0
for i in range(alpha.shape[0]):
    weight += alpha[i] * y[i] * x[i]

# plot the model: wx+b
x1 = np.min(x[:, 0])
y1 = (-bias - weight[0] * x1) / weight[1]
x2 = np.max(x[:, 0])
y2 = (-bias - weight[0] * x2) / weight[1]
ax.plot([x1, x2], [y1, y2])

# plot the support vectors
for i, alpha_i in enumerate(alpha):
    if abs(alpha_i) > 1e-3: 
        ax.scatter([x[i, 0]], [x[i, 1]], s=150, c='none', alpha=0.7,
                   linewidth=1.5, edgecolor='#AB3319')
            
plt.show()
```

 ![img](data:image/png;base64,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) 

```
def poly_ker(d): # polynomial
    def ker(x, z): 
        return (x.dot(z.T)) ** d
    return ker

def cos_ker(x, z): # cosine similarity
    return x.dot(z.T) / np.sqrt(x.dot(x.T)) / np.sqrt(z.dot(z.T))
    
def rbf_ker(sigma): # rbf kernel
    def ker(x, z):
        return np.exp(-(x - z).dot((x - z).T) / (2.0 * sigma ** 2))
    return ker
```

```
import matplotlib.pyplot as plt
from matplotlib import cm

def plot(ax, model, x, title):
    y = model(x)
    y[y < 0], y[y >= 0] = -1, 1

    category = {'+1': [], '-1': []}
    for point, label in zip(x, y):
        if label == 1.0: category['+1'].append(point)
        else: category['-1'].append(point)
    for label, pts in category.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)
    
    # plot boundary
    p = np.meshgrid(np.arange(-1.5, 1.5, 0.025), np.arange(-1.5, 1.5, 0.025))
    x = np.array([p[0].flatten(), p[1].flatten()]).T
    y = model(x)
    y[y < 0], y[y >= 0] = -1, 1
    y = np.reshape(y, p[0].shape)
    ax.contourf(p[0], p[1], y, cmap=plt.cm.coolwarm, alpha=0.4)
    
    # set title
    ax.set_title(title)

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

x, y = spiral_data()

# plot points
model_default, _, _ = svm_smo(x, y, default_ker, 1e10, 200)
plot(ax1, model_default, x, 'Default SVM')

ker = rbf_ker(0.2)
# ker = poly_ker(5)
# ker = cos_ker
model_ker, _, _ = svm_smo(x, y, ker, 1e10, 200)
plot(ax2, model_ker, x, 'SVM + RBF')

plt.show()
```

 ![](data:image/png;base64,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) 

```
from sklearn import svm
# x, y = simple_synthetic_data(50, 5, 5)
x, y = spiral_data()

model = svm.SVC(kernel='rbf', gamma=50, tol=1e-6)
model.fit(x, y)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
plot(ax, model.predict, x, 'SVM + RBF')
plt.show()
```

 ![img](data:image/png;base64,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) 


### 参考博客
[统计学习基础]()
[支持向量机SVM：原理讲解+手写公式推导+疑惑分析](https://blog.csdn.net/randompeople/article/details/90020648)
[支持向量机 - 软间隔最大化](https://blog.csdn.net/randompeople/article/details/104031825)
