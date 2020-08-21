# Enhanced LSTM for Natural Language Inference

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 
> 论文地址：https://arxiv.org/pdf/1609.06038.pdf

## 前言

- 自然语言推理（NLI: natural language inference）问题：
  - 即判断能否从一个前提p中推导出假设h
  - 简单来说，就是判断给定两个句子的三种关系：蕴含、矛盾或无关

在Query 扩召回项目中，通过各种手段挖掘出一批同义词，想对其进行流程化，所以考虑加上语义推断，作为竞赛神器 ESIM 模型，该模型在近两年横扫了好多比赛，算是 NLI (Natural Language Inference) 领域未来几年一个很难绕过的超强 baseline 了，单模型的效果可以达到 88.0% 的 Acc。

- 创新点
  - 精细的设计序列式的推断结构；
  - 考虑局部推断和全局推断。

## 模型介绍

![](img/20181213150803869.png)

- 模型结构图分为左右两边：
- 左侧就是 ESIM，
- 右侧是基于句法树的 tree-LSTM，两者合在一起交 HIM (Hybrid Inference Model)。
- 整个模型从下往上看，分为三部分：
  - input encoding；
  - local inference modeling；
  - inference composition；
  - Prediction

以 ESIM 为例

### Input Encoding

- step1 : 输入一般可以采用预训练好的词向量或者添加embedding层，这里介绍采用的是embedding层;
- step2 ：采用一个双向的LSTM，起作用主要在于对输入值做encoding，也可以理解为在做特征提取，
- step3 ：把其隐藏状态的值保留下来，

![](img/20200819085245.png)

> 其中i与j分别表示的是不同的时刻，a与b表示的是上文提到的p与h。

### Local Inference Modeling

- 目标：将上一轮 所提取到的 特征值 做 差异值 计算；
- 所用方法： Attention
- 步骤
  - s1： 计算 Attention weight（如 图 1）
  - s2： 根据attention weight计算出a与b的权重加权后的值（如 图 2）
  - s3： 得到encoding值与加权encoding值之后，下一步是分别对这两个值做差异性计算，作者认为这样的操作有助于模型效果的提升，论文有两种计算方法：
    - 对位相减
    - 对位相乘
  - s4： 把encoding两个状态的值与相减、相乘的值拼接起来（如 图 3）

![](img/20200819085713.png)
> 图 1

![](img/20200819085922.png)
> 图 2
> 注：计算 a 时，是与 b 做 加权，而非自身，b 同理

![](img/20200819090200.png)
> 图 3

### Inference Composition

在这一层中，把之前的值再一次送到了BiLSTM中，这里的BiLSTM的作用和之前的并不一样，这里主要是用于捕获局部推理信息 $m_a$ 和 $m_b$ 及其上下文，以便进行推理组合。

最后把BiLSTM得到的值进行池化操作，分别是最大池化与平均池化，并把池化之后的值再一次的拼接起来。

![](img/20200819090442.png)

### Prediction

最后把 V 送入到全连接层，激活函数采用的是tanhtanh，得到的结果送到softmax层。

## 参考

1. [文本匹配、文本相似度模型之ESIM](https://blog.csdn.net/u012526436/article/details/90380840)
2. [短文本匹配的利器-ESIM](https://zhuanlan.zhihu.com/p/47580077)