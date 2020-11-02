# 【关于 文本匹配】 那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 目录

- [【关于 文本匹配】 那些你不知道的事](#关于-文本匹配-那些你不知道的事)
  - [目录](#目录)
  - [DSSM](#dssm)
    - [动机](#动机)
    - [思路](#思路)
    - [优点](#优点)
    - [缺点](#缺点)
  - [BiMPM](#bimpm)
    - [模型介绍](#模型介绍)
  - [DIIN](#diin)
    - [模型介绍](#模型介绍-1)
  - [ESIM](#esim)
    - [模型介绍](#模型介绍-2)
  - [资料](#资料)


## DSSM

### 动机

- 问题：语义相似度问题
  - 字面匹配体现
    - 召回：在召回时，传统的文本相似性如 BM25，无法有效发现语义类 Query-Doc 结果对，如"从北京到上海的机票"与"携程网"的相似性、"快递软件"与"菜鸟裹裹"的相似性
    - 排序：在排序时，一些细微的语言变化往往带来巨大的语义变化，如"小宝宝生病怎么办"和"狗宝宝生病怎么办"、"深度学习"和"学习深度"；
  - 使用 LSA 类模型进行语义匹配，但是效果不好

### 思路

![](img/微信截图_20201027081221.png)

- 三层：
  - embedding 层对应图中的Term Vector，Word Hashing；
  - 特征提取层对应图中的，Multi-layer，Semantic feature；
  - 匹配层 Cosine similarity, Softmax；

### 优点

- 减少切词的依赖：解决了LSA、LDA、Autoencoder等方法存在的一个最大的问题，因为在英文单词中，词的数量可能是没有限制，但是字母 n-gram 的数量通常是有限的
- 基于词的特征表示比较难处理新词，字母的 n-gram可以有效表示，鲁棒性较强；
- 传统的输入层是用 Embedding 的方式（如 Word2Vec 的词向量）或者主题模型的方式（如 LDA 的主题向量）来直接做词的映射，再把各个词的向量累加或者拼接起来，由于 Word2Vec 和 LDA 都是无监督的训练，这样会给整个模型引入误差，DSSM 采用统一的有监督训练，不需要在中间过程做无监督模型的映射，因此精准度会比较高；
- 省去了人工的特征工程；

### 缺点

- word hashing可能造成冲突
- DSSM采用了词袋模型，损失了上下文信息
- 在排序中，搜索引擎的排序由多种因素决定，由于用户点击时doc的排名越靠前，点击的概率就越大，如果仅仅用点击来判断是否为正负样本，噪声比较大，难以收敛

## BiMPM

### 模型介绍

![](img/20200819125640.png)

- Word Representation Layer:其中词表示层使用预训练的Glove或Word2Vec词向量表示, 论文中还将每个单词中的字符喂给一个LSTM得到字符级别的字嵌入表示, 文中使用两者构造了一个dd维的词向量表示, 于是两个句子可以分别表示为 P:[p1,⋯,pm],Q:[q1,⋯,qn].

- Context Representation Layer: 上下文表示层, 使用相同的双向LSTM来对两个句子进行编码. 分别得到两个句子每个时间步的输出.

- Matching layer: 对两个句子PP和QQ从两个方向进行匹配, 其中⊗⊗表示某个句子的某个时间步的输出对另一个句子所有时间步的输出进行匹配的结果. 最终匹配的结果还是代表两个句子的匹配向量序列.

- Aggregation Layer: 使用另一个双向LSTM模型, 将两个匹配向量序列两个方向的最后一个时间步的表示(共4个)进行拼接, 得到两个句子的聚合表示.

- Prediction Layer: 对拼接后的表示, 使用全连接层, 再进行softmax得到最终每个标签的概率.


## DIIN

### 模型介绍

模型主要包括五层：嵌入层（Embedding Layer）、编码层（Encoding Layer）、交互层（Interaction Layer ）、特征提取层（Feature Extraction Layer）和输出层（Output Layer），如图1所示。

![](img/v2-bb3dabcdabb85d00ae394e56dd53603a_720w.jpg)

## ESIM

### 模型介绍

![](img/20181213150803869.png)

- 模型结构图分为左右两边：
- 左侧就是 ESIM，
- 右侧是基于句法树的 tree-LSTM，两者合在一起交 HIM (Hybrid Inference Model)。
- 整个模型从下往上看，分为三部分：
  - input encoding；
  - local inference modeling；
  - inference composition；
  - Prediction


## 资料

ACL2019 Simple and Effective Text Matching with Richer Alignment Features es https://github.com/alibaba-edu/simple-effective-text-matching

文本匹配相关方向打卡点总结 总结 https://www.jiqizhixin.com/articles/2019-10-18-14

ELECTRA openreview

https://openreview.net/forum?id=r1xMH1BtvB

https://github.com/Erutan-pku/LCN-for-Chinese-QA

Lattice CNNs for Matching Based Chinese Question Answering （熟悉的Lattice）

https://github.com/alibaba-edu/simple-effective-text-matching-pytorch pytorch

https://github.com/alibaba-edu/simple-effective-text-matching tf
