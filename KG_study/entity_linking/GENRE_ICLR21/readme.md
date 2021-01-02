# 【关于  Low-resource Cross-lingual Entity Linking】 那些你不知道的事

> 作者：杨夕
> 
> 论文名称：AUTOREGRESSIVE ENTITY RETRIEVAL
> 
> 论文地址：https://openreview.net/pdf?id=5k8F6UU39V
> 
> 来源：EMNLP 2020
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 目录


## 摘要

- 介绍：实体是我们表示和聚合知识的中心。例如，维基百科等百科全书是由实体构成的（例如，一篇维基百科文章）。检索给定查询的实体的能力是知识密集型任务（如实体链接和开放域问答）的基础。理解当前方法的一种方法是将分类器作为一个原子标签，每个实体一个。它们的权重向量是通过编码实体元信息（如它们的描述）产生的密集实体表示。[Entities are at the center of how we represent and aggregate knowledge.  For instance, Encyclopedias such as Wikipedia are structured by entities (e.g., one perWikipedia article). The ability to retrieve such entities given a query is fundamental for knowledge-intensive tasks such as entity linking and open-domain question answering.   One  way  to  understand  current  approaches  is  as  classifiers  amongatomic labels, one for each entity.  Their weight vectors are dense entity representations produced by encoding entity meta information such as their descriptions.]
- 缺点：
  - （i）上下文和实体的亲和力主要是通过向量点积来获取的，可能会丢失两者之间的细粒度交互[context and entity affinity is mainly captured through a vector dot product, potentially missing fine-grained interactions between the two]；
  - （ii）在考虑大型实体集时，需要大量内存来存储密集表示[a large memory foot-print is needed to store dense representations when considering large entity sets]；
  - （iii）必须在训练时对一组适当硬的负面数据进行二次抽样[an appropriately hard set of negative data has to be subsampled at training time]。
- 工作内容介绍：在这项工作中，我们提出了第一个 GENRE，通过生成其唯一的名称，从左到右，token-by-token 的自回归方式和条件的上下文。[In this work, we propose GENRE, the first system that retrieves entities by generating their unique names,left to right, token-by-token in an autoregressive fashion and conditioned on the context.  ]
- 这使得我们能够缓解上述技术问题，
  - （i）自回归公式允许我们直接捕获文本和实体名称之间的关系，有效地交叉编码两者 [the autoregressive formulation allows us to directly capture relations between context and entity name, effectively cross encoding both]；
  - （ii）由于我们的编码器-解码器结构的参数随词汇表大小而不是词汇量大小而缩放，因此内存足迹大大减少实体计数[the memory foot-print is greatly reduced because the parameters of our encoder-decoder architecture scale with vocabulary size, not entity count]；
  - （iii）准确的softmax损失可以有效地计算，而无需对负数据进行子采样[the exact softmax loss canbe efficiently computed without the need to subsample negative data.]。
- 实验结果：我们展示了该方法的有效性，在实体消歧、端到端实体链接和文档检索任务上对20多个数据集进行了实验，在使用竞争系统内存占用的一小部分的情况下，获得了最新的或非常有竞争力的结果。他们的实体，我们只需简单地指定新的名称，就可以添加




## 参考资料

1. [Facebook提出生成式实体链接、文档检索，大幅刷新SOTA！](https://mp.weixin.qq.com/s/AIHsI3L57WLqR0D5y-_BTQ)