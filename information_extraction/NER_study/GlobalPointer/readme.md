# 【关于 命名实体识别 之 W2NER 】 那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 论文：Unified Named Entity Recognition as Word-Word Relation Classification
> 
> 会议：AAAI 2022
> 
> 论文地址：https://arxiv.org/pdf/2112.10070.pdf
> 
> 代码：https://github.com/ljynlp/w2ner
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 一、摘要

So far, named entity recognition (NER) has been involved with three major types, including flat, overlapped (aka.nested), and discontinuous NER, which have mostly been studied individually. Recently, a growing interest has been built for unified NER, tackling the above three jobs concurrently with one single model. Current best-performing methods mainly include span-based and sequence-to-sequence models, where unfortunately the former merely focus on boundary identification and the latter may suffer from exposure bias. In this work, we present a novel alternative by modeling the unified NER as word-word relation classification, namely W2NER. The architecture resolves the kernel bottleneck of unified NER by effectively modeling the neighboring relations between entity words with Next-Neighboring-Word (NNW) and Tail-Head-Word-* (THW-*) relations. Based on the W2NER scheme we develop a neural framework, in which the unified NER is modeled as a 2D grid of word pairs. We then propose multi-granularity 2D convolutions for better refining the grid representations. Finally, a co-predictor is used to sufficiently reason the word-word relations. We perform extensive experiments on 14 widely-used benchmark datasets for flat, overlapped, and discontinuous NER (8 English and 6 Chinese datasets), where our model beats all the current top-performing baselines, pushing the state-of-the-art performances of unified NER.

- 动机：
  - 如何 构建解决非嵌套，嵌套，不连续实体的统一框架？
  - span-based 只关注边界识别 
  - Seq2Seq 可能会受到暴露偏差的影响
- 论文方法：
  - 通过将统一的 NER 建模为 word-word relation classification（W2NER）
  - 该架构通过使用 Next-Neighboring-Word (NNW) 和 Tail-Head-Word-* (THW-*) 关系有效地建模实体词之间的相邻关系，解决了统一 NER 的内核瓶颈。
  - 基于 W2NER 方案，我们开发了一个神经框架，其中统一的 NER 被建模为单词对的 2D 网格。
  - 然后，我们提出了多粒度 2D 卷积，以更好地细化网格表示。
  - 最后，使用一个共同预测器来充分推理词-词关系。
- 实验：在 14 个广泛使用的基准数据集上进行了广泛的实验，用于非嵌套，嵌套，不连续实体的 NER（8 个英文和 6 个中文数据集），其中我们的模型击败了所有当前表现最好的基线，推动了最先进的性能- 统一NER的mances

## 二、动机

- 如何 构建解决非嵌套，嵌套，不连续实体的统一框架？
- 序列标注： 
  - 介绍：对实体span内的每一个token进行标注，比如BIO或BIESO；
- span-based 
  - 介绍：对实体span的start和end进行标注，比如可采取指针网络、Token-pair矩阵建模、片段枚举组合预测等方式。
  - 存在问题：只关注边界识别 
- Seq2Seq 
  - 介绍：以Seq2Seq的方式进行，序列输出的文本除了label信息，Span必须出现在原文中，这就要求生成式统一建模时对解码进行限制（受限解码）
  - 存在问题：可能会受到暴露偏差的影响

## 三、论文方法

### 3.1 W2NER 介绍

- 作用：不光进行实体边界的识别，还有学习实体词之间的相邻关系。
- 两种相邻关系的学习方法:
  - Next-Neighboring-Word (NNW) : 实体词识别，表明两个词在一个实体中是否相邻(例如，实体aching和实体in是否相邻)；
  - Tail-Head-Word-* (THW- * ): 进行实体边界和类型检测，判断两个词是否分别是“*”（这里星号指实体类型）实体的尾部和头部边界(例如，症状*，尾实体legs 和 头实体aching)

![](img/微信截图_20220309133446.png)

- 效果：通过上述的两种Tag标记方式连接任意两个Word，就可以解决如上图中各种复杂的实体抽取：（ABCD分别是一个Word）
  - a): AB和CD代表两个扁平实体；
  - b): 实体BC嵌套在实体ABC中；
  - c): 实体ABC嵌套在非连续实体ABD；
  - d): 两个非连续实体ACD和BCE；

![](img/微信截图_20220309133614.png)

### 3.2 Unified NER Framework

![](img/微信截图_20220309134009.png)

#### 3.2.1 Encoder Layer

- 采用bert+bilstm，字粒度分词
- 假设输入句子长度N ，最后语义表示输出N*dh (dh为词向量维度)
  
#### 3.2.2 Convolution Layer

采取CNNs,天然适合2维网格，卷积层由3部分构成

- conditional layer with normalization：生成词对网格的表示

1. 词对的语义3维表示 V：N*N*dh;每一个词对（xi,xj）的表示为Vij
2. 由于NNW和THW关系都是单向的，所以对于词对表示（xi,xj）中xi和xj 的联合表示 暗含了 xj是以xi为条件进行表示的，所以使用CLN生产词对的语义网格，计算公式如下，将xi作为条件计算增益和偏差，根据xj计算均值和方差

![](img/微信截图_20220309134306.png)

- BERT-style grid representation：bert语义网格丰富词义信息

1. 受bert编码时token、position、sentential三种编码聚合的方式启发针对2维网格提出新的语义表示结合策略
2. 词的编码，每对单词之间的相对位置信息，在网格中区分上下三角形区域的区域信息
3. 通过MLP reduce维度，整合语义表示

- multigranularity dilated convolution：捕获近距离单词之间交互作用

利用多粒度的空洞卷积获取不同单词的作用信息

#### 3.2.3 Co-Predictor Layer

1. 采用MLP和biaffine进行关系分类
2. Biaffine Predictor：Figure3 中虚线，直接来之Encoder Layer的编码输出，相当于残差连接，在仿射结构里分别使用一个MLP来计算xi和xj的语义表示，最后用一个仿射分类器计算预测的关系分数
3. MLP Predictor：将经过Conditional Layer Normalization、BERT-Style Grid Representation Build-Up、Multi-Granularity Dilated Convolution输出的向量通过MLP进行关系分数预测
4. 最后使用softmax 计算Biaffine Predictor 和 MLP Predictor 的分数

![](img/微信截图_20220309134446.png)

#### 3.2.4 Decoding

针对四种情况的解码方式

![](img/微信截图_20220309134530.png)

#### 3.2.5 损失函数

![](img/微信截图_20220309134601.png)


## 参考

1. [算法框架-信息抽取-NER识别-Unified Named Entity Recognition as Word-Word Relation Classification（2022）](https://zhuanlan.zhihu.com/p/473742306)
2. [NER统一模型：刷爆14个中英文数据SOTA](https://zhuanlan.zhihu.com/p/476746322)