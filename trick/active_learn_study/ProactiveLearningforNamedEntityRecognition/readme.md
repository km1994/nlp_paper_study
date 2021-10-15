# 【关于 Proactive Learning for Named Entity Recognition（命名实体识别的主动学习）】 那些的你不知道的事

> 作者：杨夕 
> 
> 个人github：https://github.com/km1994/nlp_paper_study 
> 
> 论文标题：Proactive Learning for Named Entity Recognition（命名实体识别的主动学习）
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 
> 【注：手机阅读可能图片打不开！！！】


## Abstract

The goal of active learning is to minimise the cost of producing an annotated dataset, in which annotators are assumed to be perfect, i.e., they always choose the correct labels. However, in practice, annotators are not infallible, and they are likely to assign incorrect labels to some instances. Proactive learning is a generalisation of active learning that can model different kinds of annotators. Although proactive learning has been applied to certain labelling tasks, such as text classification, there is little work on its application to named entity (NE) tagging. In this paper, we propose a proactive learning method for producing NE annotated corpora, using two annotators with different levels of expertise, and who charge different amounts based on their levels of experience. To optimise both cost and annotation quality, we also propose a mechanism to present multiple sentences to annotators at each iteration. Experimental results for several corpora show that our method facilitates the construction of high-quality NE labelled datasets at minimal cost.

## 摘要

主动学习的目标是最小化生成带注释数据集的成本，其中假定注释器是完美的，即，它们总是选择正确的标签。但是，实际上，注释器并不是绝对可靠的，并且它们可能会为某些实例分配不正确的标签。主动学习是主动学习的概括，可以为不同类型的注释器建模。虽然主动学习已经应用于某些标签任务，例如文本分类，但它在应用于命名实体（NE）标记方面的工作却很少。在本文中，我们提出了一种主动学习方法，用于生成NE注释语料库，使用两个具有不同专业水平的注释器，并根据他们的经验水平收取不同的数量。为了优化成本和注释质量，我们还提出了一种机制，在每次迭代时向注释器呈现多个句子。几个语料库的实验结果表明，我们的方法以最低的成本促进了高质量NE标记数据集的构建。
