# 【关于 SpanBERT 】 那些的你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> NLP 面经地址：https://github.com/km1994/NLP-Interview-Notes
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 
> 论文：SpanBERT: Improving Pre-training by Representing and Predicting Spans
> 
> 发表会议：
> 
> 论文地址：https://arxiv.org/abs/1907.10529
> 
> github：https://github.com/facebookresearch/SpanBERT

## 摘要

- 动机：旨在更好地表示和预测文本的 span;
- 论文方法->扩展了BERT：
  - （1）屏蔽连续的随机跨度，而不是随机标记；
  - （2）训练跨度边界表示来预测屏蔽跨度的整个内容，而不依赖其中的单个标记表示。
- 实验结果：
  - SpanBERT始终优于BERT和我们更好调整的基线，在跨选择任务（如问题回答和共指消解）上有实质性的收益。特别是在训练数据和模型大小与BERT-large相同的情况下，我们的单一模型在1.1班和2.0班分别获得94.6%和88.7%的F1。我们还实现了OntoNotes共指消解任务（79.6\%F1）的最新发展，在TACRED关系抽取基准测试上表现出色，甚至在GLUE上也有所提高。



## 参考资料

1. [SpanBERT：提出基于分词的预训练模型，多项任务性能超越现有模型！](https://cloud.tencent.com/developer/article/1476168)
2. [解读SpanBERT:《Improving Pre-training by Representing and Predicting Spans》](https://blog.csdn.net/weixin_37947156/article/details/99210514)
3. [NLP中的预训练语言模型（二）—— Facebook的SpanBERT和RoBERTa](https://www.shuzhiduo.com/A/Gkz1MGQZzR/)

