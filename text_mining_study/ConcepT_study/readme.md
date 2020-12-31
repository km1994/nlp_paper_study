# 【关于 ConcepT】那些你不知道的事

> 笔者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 论文：A User-Centered Concept Mining System for Query and Document Understanding at Tencent
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 目录

- [【关于 ConcepT】那些你不知道的事](#关于-concept那些你不知道的事)
  - [目录](#目录)
  - [内容](#内容)
    - [概念是什么?](#概念是什么)
    - [论文动机是什么？](#论文动机是什么)
    - [ConcepT概念挖掘系统的思路？](#concept概念挖掘系统的思路)
    - [ConcepT的贡献点是什么？](#concept的贡献点是什么)
    - [ConcepT采用的方法什么？](#concept采用的方法什么)
  - [参考](#参考)


## 内容

### 概念是什么?

- 介绍："概念"是事物的抽象；
- 作用：认识“概念”（concept）是人类认识世界的重要基石；
- 应用：对于自然语言理解，提取概念（extract concept）和对文本进行概念化（conceptualization）是至关重要的研究问题。
- 举例：
  - 本田思域（Honda Civic）/现代伊兰特（Hyundai Elantra）-联想-> 油耗低的车(fuel-efficient cars)/经济型车(economy cars)-联想-> 福特福克斯（Ford Focus）/尼桑骐达（Nissan Versa）
  
![](img/微信截图_20200905200354.png)

### 论文动机是什么？

1. Query短且不规范，传统的Hearst pattern不管用；
2. 人工标注过于主观，无法捕获用户兴趣及意图；
3. 一般关键词抽取方法只能抽取连续字符串，而用户感兴趣的概念可能是非连续字符串；
4. 传统方法时效性较差

### ConcepT概念挖掘系统的思路？

1. 用以提取符合用户兴趣和认知粒度的概念；
2. ConcepT系统从大量的用户query搜索点击日志中提取概念；
3. 并进一步将主题、概念和实体联系在一起，构成一个分层级的认知系统。

### ConcepT的贡献点是什么？

1. 提出两种无监督模型，bootstrapping和query-title alignment，从大量搜索日志中提取出以用户为中心（user-centered）的概念；
2. 基于以上策略提取的种子概念，进一步训练有监督模型（条件随机场CRF + 分类器）来从query和点击title中进一步提取概念短语；
3. 提出了两种策略来对长文章打上概念标签，丰富对文章主题的刻画；
4. 通过提取主题、概念、实体之间的 isA关系，我们构建了一个三层的分级系统，来保存它们之间的联系。
5. 实验证明，ConcepT系统能精确地从query中提取高质量的概念短语，以及将长文章打上相关的概念标签。在线A/B test证明，ConcepT系统能相对提升6.01%的信息流曝光效率。

### ConcepT采用的方法什么？

## 参考

1. [腾讯提出概念挖掘系统ConcepT-学习笔记](https://zhuanlan.zhihu.com/p/85494010)