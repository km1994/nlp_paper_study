# 【关于 Double Graph Based Reasoning for Document-level Relation Extraction】 那些的你不知道的事

> 作者：杨夕
> 
> 论文：Double Graph Based Reasoning for Document-level Relation Extraction
> 
> 论文地址：https://arxiv.org/abs/2009.13752
> 
> github：https://github.com/DreamInvoker/GAIN

## 目录

- [【关于 Double Graph Based Reasoning for Document-level Relation Extraction】 那些的你不知道的事](#关于-double-graph-based-reasoning-for-document-level-relation-extraction-那些的你不知道的事)
  - [目录](#目录)
  - [背景](#背景)
  - [背景](#背景-1)

## 背景

文档级关系抽取是为了抽取文档中实体之间的关系。与句子级关系抽取不同，它需要对文档中对多个句子进行推理。文章**提出了一种具有双图特征的图聚合推理网络(GAIN)**。GAIN首先构建了一个**异构的提及图**(hMG)来建模文档中不同提及(mention)之间的复杂交互。同时还**构造了实体级图**(EG)，并在此基础上提出了一种新的**路径推理机制**来推断实体之间的关系。在公共数据集DocRED上进行实验，实验结果表明GAIN显著优于之前的技术，比之前的冠军模型在F1上高出2.85。

## 背景

许多事实知识隐藏信息




