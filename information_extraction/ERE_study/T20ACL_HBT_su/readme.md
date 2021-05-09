# 【关于 HBT】 那些的你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 论文名称：A Novel Hierarchical Binary Tagging Framework for Relational Triple Extraction
> 
> 【注：手机阅读可能图片打不开！！！】
> 

## 摘要

Extracting relational triples from unstructured text is crucial for large-scale knowledge graph construction. However, few existing works excel in solving the overlapping triple problem where multiple relational triples in the same sentence share the same entities. In this work, we introduce a fresh perspective to revisit the relational triple extraction task and propose a novel Hierarchical Binary Tagging (HBT) framework derived from a principled problem formulation. Instead of treating relations as discrete labels as in previous works, our new framework models relations as functions that map subjects to objects in a sentence, which naturally handles the overlapping problem. Experiments show that the proposed framework already outperforms state-of-the-art methods even when its encoder module uses a randomly initialized BERT encoder, showing the power of the new tagging framework. It enjoys further performance boost when employing a pretrained BERT encoder, outperforming the strongest baseline by 17.5 and 30.2 absolute gain in F1-score on two public datasets NYT and WebNLG, respectively. In-depth analysis on different scenarios of overlapping triples shows that the method delivers consistent performance gain across all these scenarios.

从非结构化文本中提取关系三元组对于大规模知识图的构建至关重要。

但是，很少有现有的著作能很好地解决重叠三重问题，在该问题中，同一句子中的多个关系三重共享同一实体。

在这项工作中，我们引入了一个新的视角来重新审视关系三重提取任务，并提出了一个从有原则的问题表达中衍生出来的新颖的分层二进制标记（HBT）框架。我们的新框架没有像以前的作品那样将关系视为离散标签，而是将关系建模为将主语映射到句子中的宾语的函数，从而自然地解决了重叠问题。

实验表明，即使在其编码器模块使用随机初始化的BERT编码器的情况下，所提出的框架也已经超越了最新方法，显示了新标签框架的强大功能。当使用预训练的BERT编码器时，它在性能上得到了进一步的提升，在两个公共数据集NYT和WebNLG上，F1评分的绝对增益分别比最强的基线高17.5和30.2。对重叠三元组的不同方案的深入分析表明，该方法在所有这些方案中均提供了一致的性能提升。

## 一、引言

### 1.1 背景知识

关系三元组抽取(Relational Triple Extraction, RTE)，也叫实体-关系联合抽取，是信息抽取领域中的一个经典任务，三元组抽取旨在从文本中抽取出结构化的关系三元组(Subject, Relation, Object)用以构建知识图谱。

近年来，随着NLP领域的不断发展，在简单语境下(例如，一个句子仅包含一个关系三元组)进行关系三元组抽取已经能够达到不错的效果。但在复杂语境下(一个句子中包含多个关系三元组，有时甚至多达五个以上)，尤其当多个三元组有重叠的情况时，许多现有模型的表现就显得有些捉襟见肘了。

### 1.2 之前方法介绍

#### 1.2.1 pipeline approach

##### 1.2.1.1 思路

pipeline approach 方法的核心就是将 实体-关系联合抽取任务 分成 实体抽取+关系分类 两个任务，思路如下：

1. 实体抽取：利用一个命名实体识别模型 识别句子中的所有实体；
2. 关系分类：利用 一个关系分类模型 对每个实体对执行关系分类。 【这一步其实可以理解为文本分类任务，但是和文本分类任务的区别在于，关系分类不仅需要学习句子信息，还要知道 实体对在 句子中 位置信息】
   
##### 1.2.1.2 问题

- 误差传递问题：由于 该方法将 实体-关系联合抽取任务 分成 实体抽取+关系分类 两个任务处理，所以 实体抽取任务的错误无法在后期阶段进行纠正，因此这种方法容易遭受错误传播问题；

#### 1.2.2 feature-based models and neural network-based models 

#####  1.2.2.1 思路

通过用学习表示替换人工构建的特征，基于神经网络的模型在 关系三元组 提取 任务中取得了相当大的成功。

##### 1.2.2.2 问题

- 实体关系重叠问题：大多数现有方法无法正确处理句子包含多个相互重叠的关系三元组的情况。

<img src="img/1.png" width=700>

> 图 1 中介绍了三种 关系三元组 场景 <br/>
> Normal 关系。表示三元组之间无重叠； (United states ，Trump) 之间的 关系为 Country_president，（Tim Cook，Apple Inc） 之间的关系为 Company_CEO；这种 三元组关系 比较简单 <br/>
> EPO(Entity Pair Overlap)。表示多（两）个三元组之间共享同一个实体对；（IQuentin Tarantino，Django Unchained） 实体对 间 存在 Act_in 和 Direct_movic 两种关系。 <br/>
>  SEO(Single Entity Overlap)。表示多（两）个三元组之间仅共享一个实体； （Jackie，Birth, Wachinghua） 和 （Wachinghua，Capital， United States） 共享 实体 Wachinghua。

#### 1.2.3 基于Seq2Seq模型  and GCN

##### 1.2.3.1 思路

Zeng 是最早在关系三重提取中考虑重叠三重问题的人之一。 他们介绍了如图 1 所示的不同重叠模式的类别，并提出了具有复制机制以提取三元组的序列到序列（Seq2Seq）模型。 他们基于Seq2Seq模型，进一步研究了提取顺序的影响，并通过强化学习获得了很大的改进。 

Fu 还通过使用基于图卷积网络（GCN）的模型将文本建模为关系图来研究重叠三重问题。

##### 1.2.3.2 问题

- 过多 negative examples：在所有提取的实体对中，很多都不形成有效关系，从而产生了太多的negative examples；
- EPO(Entity Pair Overlap) 问题：当同一实体参与多个关系时，分类器可能会感到困惑。 没有足够的训练样例的情况下，分类器就很难准确指出实体参与的关系；


## 二、论文工作

- 方式：实现了一个不受重叠三元组问题困扰的HBT标注框架(Hierarchical Binary Tagging Framework)来解决RTE任务；
- 核心思想：把关系(Relation)建模为将头实体(Subject)映射到尾实体(Object)的函数，而不是将其视为实体对上的标签。

论文并不是学习关系分类器f（s，o）→r，而是学习关系特定的标记器fr（s）→o；每个标记器都可以识别特定关系下给定 subject 的可能 object(s)。 或不返回任何 object，表示给定的主题和关系没有 triple。

- 思路：
  - 首先，我们确定句子中所有可能的 subjects； 
  - 然后针对每个subjects，我们应用特定于关系的标记器来同时识别所有可能的 relations 和相应的 objects。

![](img/微信截图_20210509162308.png)

## 三、CASREl 结构介绍

### 3.1 BERT Encoder层

![](img/微信截图_20210509163447.png)

这里使用 Bert 做 Encoder，其实就是 用 Bert 做 Embedding 层使用。

### 3.2 Hierarchical Decoder层

Hierarchical Decoder 层 由两部分组成：

1. Subject tagger 层：用于 提取 Subject;
2. Relation-Specific Object Taggers 层：由一系列relation-specific object taggers（之所以这里是多个taggers是因为有多个可能的relation）；

#### 3.2.1 Subject Tagger 层

![](img/微信截图_20210509165814.png)

- 目标：检测 Subject 的开始和结束位置
- 方法：利用两个相同的 二分类器，来检测 每个 Subject 的开始和结束位置；
- 做法：

对BERT的输出的特征向量作sigmoid激活，计算该token作为subject的开始、结束的概率大小。如果 概率 超过设定阈值，则标记为1，反之为0。

![](img/微信截图_20210509170248.png)

> 其中xi是第i个token的编码表示；pi是第i个token是subject的start或者end的概率

为了获得更好的W（weight）和b（bias）subject tagger需要优化这个似然函数：

![](img/微信截图_20210509170450.png)

#### 3.2.2 Relation-specific Object Taggers层

![](img/微信截图_20210509170733.png)

- 目标：检测 Object 的开始和结束位置
- 方法：利用两个相同的 二分类器，来检测 每个 Object 的开始和结束位置，但是 Relation-specific Object Taggers层 需要 融入上一步的 subject 特征，结合之前BERT Encoder的编码内容，用来在指定的relation下预测对应的object的起止位置，概率计算如下和之前相比多了v：

![](img/微信截图_20210509171143.png)

Suject Tagger预测的第k个实体的平均向量，如

![](img/微信截图_20210509171245.png)

这么做的目的是保证xi和v是相同的维度

对于每个关系r对应的tagger，需要优化的似然函数如下来获得更好的W（weight）和b（bias）这个公式等号右边和之前是完全一样的：

![](img/微信截图_20210509171341.png)

### 3.3 损失函数

![](img/微信截图_20210509171535.png)

1. 公式 1 ： 对于training set D上的sentence xj和xj中可能存在的三元组的集合 Tj， 利用公式 1 去最大化data likelihood；
2. 公式 2 ： 采用 链式法则 将第一个公式 转化为 第二个公式；

> 右边部分下角标的 表示 Tj中指定s的三元组集合，集合中的ro对来计算后面这个部分

3. 公式 3 ：对于给定的一个subject，其在句子中所参与的关系个数一般来说是有限的，因此只有部分relation能够将其映射到相应的object上去(对应公式3的中间部分)，最终得到一个有效的三元组。

> 注：对于未参与的关系，文中提出了”null object”的概念，也就是说，在这种情况下函数会将subject映射到一个空的尾实体上(对应公式3的右端部分)，表示subject并不参与该关系，也就无法抽取出有效的三元组。

4. 损失函数：

![](img/微信截图_20210509172205.png)


## 贡献

1. We introduce a fresh perspective to revisit the relational triple extraction task with a principled problem formulation, which implies a general algorithmic framework that addresses the overlapping triple problem by design. 

2. We instantiate the above framework as a novel hierarchical binary tagging model on top of a Transformer encoder. This allows the model to combine the power of the novel tagging framework with the prior knowledge in pretrained large-scale language models. 

3. Extensive experiments on two public datasets show that the proposed framework overwhelmingly outperforms state-of-the-art methods, achieving 17.5 and 30.2 absolute gain in F1-score on the two datasets respectively. Detailed analyses show that our model gains consistent improvement in all scenarios. 

## 结论

In this paper, we introduce a novel hierarchical binary tagging (HBT) framework derived from a principled problem formulation for relational triple extraction. Instead of modeling relations as discrete labels of entity pairs, we model the relations as functions that map subjects to objects, which provides a fresh perspective to revisit the relational triple extraction task. As a consequent, our model can simultaneously extract multiple relational triples from sentences, without suffering from the overlapping problem. We conduct extensive experiments on two widely used datasets to validate the effectiveness of the proposed HBT framework. Experimental results show that our model overwhelmingly outperforms state-of-theart baselines over different scenarios, especially on the extraction of overlapping relational triples.

在本文中，我们介绍了一种新颖的层次化二进制标记（HBT）框架，该框架源自用于关系三重提取的原则性问题公式。 我们不是将关系建模为实体对的离散标签，而是将关系建模为将主题映射到对象的函数，这为重新审视关系三重提取任务提供了新的视角。 因此，我们的模型可以同时从句子中提取多个关系三元组，而不会出现重叠问题。 我们对两个广泛使用的数据集进行了广泛的实验，以验证所提出的HBT框架的有效性。 实验结果表明，在不同情况下，特别是在提取重叠的关系三元组时，我们的模型绝对优于最新的基准。


## 参考

1. [A Novel Hierarchical Binary Tagging Framework for Relational Triple Extraction](https://xiaominglalala.github.io/2020/05/01/A-Novel-Hierarchical-Binary-Tagging-Framework-for/)
2. [论文笔记：A Novel Cascade Binary Tagging Framework for Relational Triple Extraction](https://zhuanlan.zhihu.com/p/360354799)