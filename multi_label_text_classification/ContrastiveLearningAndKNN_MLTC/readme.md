# 【关于 对比学习+KNN for Multi-label Text Classification 】 那些你不知道的事

> 作者：杨夕
> 
> 论文：Contrastive Learning-Enhanced Nearest Neighbor Mechanism for Multi-Label Text Classification
> 
> 论文地址：https://aclanthology.org/2022.acl-short.75.pdf
> 
> github: 
> 
> 论文出处：ACL2022
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> NLP 面经地址：https://github.com/km1994/NLP-Interview-Notes
> 
> 推荐系统 百面百搭：https://github.com/km1994/RES-Interview-Notes
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。


## 一、引言

Multi-Label Text Classification (MLTC) is a fundamental and challenging task in natural language processing. Previous studies mainly focus on learning text representation and modeling label correlation. However, they neglect the rich knowledge from the existing similar instances when predicting labels of a specific text. To address this oversight, we propose a k nearest neighbor (kNN) mechanism which retrieves several neighbor instances and interpolates the model output with their labels. Moreover, we design a multi-label contrastive learning objective that makes the model aware of the kNN classification process and improves the quality of the retrieved neighbors during inference. Extensive experiments show that our method can bring consistent and considerable performance improvement to multiple MLTC models including the state-of-the-art pretrained and non-pretrained ones.



## 二、动机



## 参考

1. [Contrastive Learning-Enhanced Nearest Neighbor Mechanism for Multi-Label Text Classification](https://aclanthology.org/2022.acl-short.75.pdf)
2. [ACL'22 | 使用对比学习增强多标签文本分类中的k近邻机制](https://blog.csdn.net/qq_27590277/article/details/123367035)
3. [对比学习+KNN来提升多标签文本分类任务](https://zhuanlan.zhihu.com/p/567899257)

