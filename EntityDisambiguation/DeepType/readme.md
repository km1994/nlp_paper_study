
# 【关于 DeepType】 那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 论文：DeepType: Multilingual Entity Linking by Neural Type System Evolution
> 
> 论文地址：https://arxiv.org/abs/1802.01021
> 
> github：https://github.com/openai/deeptype
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 目录



## 摘要

- 动机：
  - 整合结构化数据和非结构化数据，涉及到许多关于如何最好地重新发送信息以使其被捕获或有用的决策，并手工标记大量数据。
- 思路：通过将符号信息显式地集成到具有类型系统的神经网络的推理过程中
- 方法:提出了一个两步算法：
  - 1）基于离散变量的启发式搜索或随机优化，定义一个由预言机通知的类型系统和一个可学习性启发式算法；
  - 2）梯度下降以适应分类器参数。
- 效果：我们将DeepType应用于三个标准数据集（即WikiDisamb30、CoNLL（YAGO）、TAC KBP 2010）上的实体链接问题，发现DeepType在很大程度上优于所有现有的解决方案，包括依赖于人类设计的类型系统或最近基于深度学习的实体嵌入的方法，而明确地使用符号信息可以让它无需再培训就可以集成新的内容。

## 前言

- 现有方法存在问题：
  - 1、根据目标任务的实用程序或信息增益选择正确的符号信息；
  - 2、设计符号信息的表示(层次、语法、约束)；
  - 3、手工标记大量数据；
- DeepType 方法：通过将符号信息显式地集成到神经网络的推理过程中，克服了这些困难
- 两步算法：
  - 1、对离散变量赋值控制型系统设计进行启发式搜索或随机优化，使用Oracle和可学习性启发式算法，保证设计决策易于神经网络学习，并对目标任务进行改进。
  - 2、梯度下降来拟合分类器参数来预测类型系统的行为












## 读书笔记：
- Discovering Types for Entity Disambiguation ： https://openai.com/blog/discovering-types-for-entity-disambiguation/
- 读《DeepType: Multilingual Entity Linking by Neural Type System Evolution》：https://blog.csdn.net/Weiruimolv/article/details/89481205
- 知识图谱 | 实体链接：https://zhuanlan.zhihu.com/p/81073607
- DeepType剖析，以及如何使用DeepType完成实体链接：https://blog.csdn.net/Real_Brilliant/article/details/86601593