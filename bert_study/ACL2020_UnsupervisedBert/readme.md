# Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT

> 作者：杨夕
> 
> 论文链接：https://arxiv.org/pdf/2004.14786.pdf
> 
> 代码链接：https://github.com/bojone/perturbed_masking
> 
> 【注：手机阅读可能图片打不开！！！】

## 摘要

By introducing a small set of additional parameters, a probe learns to solve specific linguistic tasks (e.g., dependency parsing) in a supervised manner using feature representations (e.g., contextualized embeddings). The effectiveness of such probing tasks is taken as evidence that the pre-trained model encodes linguistic knowledge. 

However, this approach of evaluating a language model is undermined by the uncertainty of the amount of knowledge that is learned by the probe itself. Complementary to those works, we propose a parameter-free probing technique for analyzing pre-trained language models (e.g., BERT). Our method does not require direct supervision from the probing tasks, nor do we introduce additional parameters to the probing process. 

Our experiments on BERT show that syntactic trees recovered from BERT using our method are significantly better than linguistically-uninformed baselines. We further feed the empirically induced dependency structures into a downstream sentiment classification task and find its improvement compatible with or even superior to a human-designed dependency schema.

通过引入少量的附加参数，probe learns 在监督方式中使用特征表示（例如，上下文嵌入）来 解决特定的语言任务（例如，依赖解析）。这样的probe  tasks 的有效性被视为预训练模型编码语言知识的证据。

但是，这种评估语言模型的方法会因 probe 本身所学知识量的不确定性而受到破坏。作为这些工作的补充，我们提出了一种无参数的 probe 技术来分析预训练的语言模型（例如BERT）。我们的方法不需要 probe 任务的直接监督，也不需要在 probing 过程中引入其他参数。

我们在BERT上进行的实验表明，使用我们的方法从BERT恢复的语法树比语言上不了解的基线要好得多。我们进一步将根据经验引入的依存关系结构输入到下游的情感分类任务中，并发现它的改进与人工设计的依存关系兼容甚至更好。

## 介绍

probe 是简单的神经网络（具有少量附加参数），其使用预先训练的模型（例如，隐藏状态激活，注意权重）生成的特征表示，并经过训练以执行监督任务（例如，依赖性 标签）。 假设所测得的质量主要归因于预先训练的语言模型，则使用 probe 的性能来测量所生成表示的质量。

## 方法介绍

- Perturbed Masking 
  - 介绍：parameter-free probing technique
  - 目标：analyze and interpret pre-trained models，测量一个单词xj对预测另一个单词xi的影响，然后从该单词间信息中得出全局语言属性（例如，依赖树）。




## 参考资料

1. [ACL2020 | 无监督？无监督！你没试过的BERT的全新用法](https://aminer.cn/research_report/5f0d6f7b21d8d82f52e59dab)
2. [无监督分词和句法分析！原来BERT还可以这样用  科学空间](https://kexue.fm/archives/7476)
