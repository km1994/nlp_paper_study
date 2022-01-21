# 【关于 MacBERT 】那些你不知道的事

> 作者：杨夕
>
> 论文名称：Revisiting Pre-trained Models for Chinese Natural Language Processing 
> 
> 会议：EMNLP 2020
>
> 论文地址：https://arxiv.org/abs/2004.13922
> 
> 论文源码地址：https://github.com/ymcui/MacBERT
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。


## 一、动机

主要为了解决与训练阶段和微调阶段存在的差异性


## 二、解决方法

### 2.1 MLM

1. 使用Whole Word Masking、N-gram Masking：single token、2-gram、3-gram、4-gram分别对应比例为0.4、0.3、0.2、0.1；
2. 由于finetuning时从未见过[MASK]token，因此使用相似的word进行替换。使用工具Synonyms toolkit 获得相似的词。如果被选中的N-gram存在相似的词，则随机选择相似的词进行替换，否则随机选择任意词替换；
3. 对于一个输入文本，15%的词进行masking。其中80%的使用相似的词进行替换，10%使用完全随机替换，10%保持不变。

### 2.2 NSP

采用ALBERT提出的SOP替换NSP


## 参考

1. [Revisiting Pre-trained Models for Chinese Natural Language Processing ](https://arxiv.org/abs/2004.13922)
2. [MacBERT 的改进（Revisiting Pre-Trained Models for Chinese Natural Language Processing）](https://blog.csdn.net/weixin_40122615/article/details/109317504)
3. [学习笔记：Revisiting Pre-trained Models for Chinese Natural Language Processing](https://zhuanlan.zhihu.com/p/354664711)
4. [MacBERT：MLM as correction BERT](https://zhuanlan.zhihu.com/p/250595837)



