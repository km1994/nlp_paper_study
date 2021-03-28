# 【关于 DAAT 】 那些的你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> NLP 面经地址：https://github.com/km1994/NLP-Interview-Notes
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 
> 论文：Coupling Distant Annotation and Adversarial Training for Cross-Domain Chinese Word Segmentation
> 
> 发表会议：ACL2020
> 
> 论文地址：hhttps://arxiv.org/abs/2007.08186
> 
> github：https://github.com/Alibaba-NLP/DAAT-CWS


## 一、论文摘要

- 动机：完全监督的神经方法在中文分词（CWS）的任务中取得了重大进展。将监督模型应用于域外数据时，其性能往往会急剧下降。
- 性能下降原因：
  - 跨域的分布差距
  - 词汇不足（OOV）问题。
- 论文方法：本文提出将跨域CWS的远距离注释和对抗训练相结合。
  - 对于 Distant annotation（DA）远程注释，我们重新考虑了“汉语单词”的本质，并设计了一种自动的远程注释机制，该机制不需要目标域的任何监督或预定义词典。该方法可以有效地探索特定领域的单词并远距离注释目标领域的原始文本。
  - 对于Adversarial Training（AT）对抗训练，我们开发了一个句子级训练程序来执行降噪和源域信息的最大利用。
- 实验结果：在跨多个域的多个真实世界数据集上进行的实验表明，我们的模型具有优越性和鲁棒性，大大优于以前的最新跨域CWS方法。

## 二、方法介绍

### 2.1 Distant annotation（DA）

- 目的：自动生成目标域内句子的分词结果
- 方法：是在不需要任何人工定义词典的情况下，自动对目标领域文本实现自动标注。
- 思路：
  - 基本分词器：使用来自源域的标注数据训练，用于识别源域和目标域中常见的单词
  - 特定领域的单词挖掘器：旨在探索目标特定于领域的单词
- 存在问题
  - 存在影响最终性能的标注错误问题

### 2.2 Adversarial Training（AT）

- 动机：为了降低噪声数据的影响，更好地利用源域数据，
- 方法：在源域数据集和通过Distant annotation构造的目标领域数据集上联合进行Adversarial training的方法。
- 优点：Adversarial training模块可以捕获特定领域更深入的特性，和不可知领域的特性。


## 参考资料

1. [Coupling Distant Annotation and Adversarial Training for Cross-Domain Chinese Word Segmentation](https://www.x-mol.com/paper/1284222590381637632?adv)
2. [中文分词学习笔记](https://blog.csdn.net/zerozzl01/article/details/109254512)