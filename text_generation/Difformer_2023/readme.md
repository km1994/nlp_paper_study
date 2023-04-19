# Difformer - 在嵌入空间上增强扩散模型来做文本生成

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> NLP 面经地址：https://github.com/km1994/NLP-Interview-Notes
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 
> 论文： Difformer:  empowering diffusion models on the embedding space for text generation
> 
> 发表会议：
> 
> 论文地址：https://arxiv.org/pdf/2212.09412.pdf
> 
> github：
> 

## 一、动机

问题：“连续数据空间”和“嵌入空间”之间有挑战存在。

continuous data space vs. embedding space。

- 问题一，embeddings的data distribution是可学习的，这可能会导致Loss function崩溃；
- 问题二，常用词和偏僻词的embeddings向量的范数(norm）是有区别的，如果追加同样的噪声，会得到次优的结果。
- 问题三，作者发现，normal level of noise （普通水平下的噪声）会导致模型的训练不充分问题。

## 二、论文方法

面对上面几个挑战，作者提出了：

Difformer，diffusion + transformer.

包括三个核心模块：

1. 一个额外的锚损失函数，anchor loss function，来稳定训练过程；
2. 一个面向embedding的layer normalization，放到embedding layer之上，来对常用词和偏僻词的嵌入进行归一化到一个uniform scale（统一的刻度，统一的尺寸），从而可以消除他们的多尺度的影响；
3. 一个给高斯噪声的噪声因子，noise factor，来提升增加的高斯噪声的刻度，来提升每一个扩散步上的，去噪目标的指导。

效果：在两个代表性的文本生成任务上：机器翻译和摘要生成上，difformer比其他embedding diffusion baselines效果更好。

## 参考

1. [[论文尝鲜]Difformer - 在嵌入空间上增强扩散模型来做文本生成](https://zhuanlan.zhihu.com/p/616854466)

