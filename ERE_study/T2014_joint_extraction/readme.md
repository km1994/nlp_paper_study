# Incremental Joint Extraction of Entity Mentions and Relations

## 摘要

We present an incremental joint framework to simultaneously extract entity mentions and relations using structured perceptron with efficient beam-search. A segment-based decoder based on the idea of semi-Markov chain is adopted to the new framework as opposed to traditional token-based tagging. In addition, by virtue of the inexact search, we developed a number of new and effective global features as soft constraints to capture the interdependency among entity mentions and relations. Experiments on Automatic Content Extraction (ACE)1 corpora demonstrate that our joint model significantly outperforms a strong pipelined baseline, which attains better performance than the best-reported end-to-end system. 

提出了一种增量联合框架，利用结构感知器和有效的集束搜索同时提取提及的实体和关系。新框架采用了基于半马尔可夫链思想的基于分段的解码器，与传统的基于标记的标记方法不同。此外，通过不精确搜索，我们开发了一些新的和有效的全局特性作为软约束来捕获提及的实体和关系之间的相互依赖性。在自动内容提取(ACE)1语料库上的实验表明，我们的联合模型显著优于强流水线方法基线，该基线的性能优于最佳端到端系统。


## 参考资料

1. [实体-关系联合抽取：Incremental Joint Extraction of Entity Mentions and Relations](https://blog.csdn.net/MaybeForever/article/details/102972330)