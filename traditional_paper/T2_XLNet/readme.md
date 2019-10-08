# XLNet Generalized Autoregressive Pretraining for Language Understanding

## 摘要

由于具有双向上下文建模的能力，像BERT这样基于自动去噪的预训练语言模型比基于自回归的预训练语言模型的性能更好。<font color=#f0f size=4 face="黑体">然而，依赖于使用带掩码损坏的输入，BERT忽略了掩码位置之间的依赖性，进而受到了预训练-微调不一致的影响。</font>根据这些优点和缺点，我们提出了XLNet，一种广义自回归预训练方法，它（1）通过最大化输入序列的因式分解的所有排列的似然函数的期望来学习双向上下文，并且（2）由于其自回归方法，克服了BERT的局限性。此外，XLNet将最先进的自回归模型Transformer-XL的思想整合到预训练中。实验表明，XLNet在20个任务上常大幅度优于BERT的表现，并在18个任务中实现最先进的结果，包括问答、自然语言推理、情感分析和文档排名。


## 动机

文章从AR(autoregressive，自回归)和AE(autoencoding，自编码)的角度出发，解释论文动机。

AR LM，即自回归语言模型。具体而言，给定一个序列，当前token时刻只知道前面的信息，而不知道后面的信息，即使分成正向和反向计算当前token时刻的概率分布，也是同样的原则，ELMo、GPT是属于这个范畴。对于一些自然语言理解任务而言，是给定上下文的，即使ELMo把两个的方向计算的信息concat，但也是独立计算，对上下文的编码是有缺陷的。

AE LM，即自编码语言模型。BERT通过预测原始数据里MASK掉的token来预训练语言模型，预测[MASK]使用了上下文信息，弥补了AR LM的缺陷。但是[MASK]只在预训练的时候用到，finetune的时候是不用的，这使得pretrain/train不一致。并且，BERT假定每个[MASK]与其他[MASK]是相互独立的，不能计算序列、长期依赖的联合概率。即使BERT的NSP预训练任务一定程度上给了模型建模句间关系的能力，但是还是对长文本不敏感。

## 论文思路

本文结合AR LM和AE LM，在Transformer-XL的基础上提出generalized autoregressive method，XLNet。

- 所有的分解序列作为一个集合，从中采样一个序列，XLNet按照AR LM的计算方式最大化有关序列所有可能的因式分解的排列的对数似然函数的期望。通常，当前token的上文包含left和right的tokens：比如原始序列为1-2-3-4，分解序列中采样一个为2-4-1-3，那么如果当前token为3，XLNet的方式就可以看到所有的信息，当然这也是理想情况。
- 引入Transformer-XL的segment recurrence mechanism和relative encoding scheme。
- 引入Masked Two-Stream Self-Attention解决PLM出现的目标预测歧义【the ambiguity in target prediction】问题。举个例子，比如分解序列中采样一个为2-4-6-1-3-5的序列，假设要预测[1]的token，按照经典的Transformer来计算next-token的概率分布，位置[1]token的概率就是通过[2,4,6]位置上的tokens来计算softmax，不会把[1]作为输入来计算的。但是如果以这种方式去预测next-token，这对[3,5]的预测就会产生影响，因为如果[1]的预测出现错误会把错误传给后面。对后面每一个token的预测，需要建立在之前token都已知的条件下。因此本文计算了两个self-attention计算方式，一个mask当前词，attention值记为$g$；一个已知当前词，attention值记为$h$。最后假设self-attention一共有$M$层，用第$M$层、$t$时刻的$g_t$，去预测词$x_t$。

## Model 

### Permutation Language Modeling

首先代码会根据输入序列的长度采样一个排列，然后用Transformer中attention mask的方式实现排列，如果原始序列长度为T，那么理论上一共有T的阶乘种情况。PLM的目标函数就是所有排列情况（论文里设定：统共T种）的期望最大：

$$
\max _{\theta} \quad \mathbb{E}_{\mathbf{Z} \sim \mathcal{Z}_{T}}\left[\sum_{t=1}^{T} \log p_{\theta}\left(x_{z_{t}} | \mathbf{x}_{\mathbf{z}<t}\right]\right]
$$

这样pretrain和finetune阶段就一样了，输入都是原始序列，通过attention mask实现随机产生的排列。下图是排列语言模型的表现形式：

![Permutation Language Modeling](img/plm.png)

> 注：假设要预测t=3的词，按照不同的排列顺序，h_3的上文都不一样，用attention-mask的方式得到t=3的上文。

### Two-Stream Self-Attention for Target-Aware Representations

上面是构造输入，这里就是自回归地得到每一时刻的概率分布，示意图如下：

![](img/two_attention.png)

动机部分已经介绍过为什么要计算两个self-attention。

(a)代表context stream self-attention，以[1,t]时刻的词作为K、V，t时刻的词作为Q计算当前词的信息，把排列之后的原始序列信息用h记忆起来。

(b)代表query stream self-attention，mask掉当前词，以[1,t-1]时刻的词作为K、V，t时刻的词作为Q预测当前词，得到概率分布。

(c)代表通过多层的masked two-stream attention，最后用t时刻的$g_t$来预测x_t。

计算公式如下：

$$
\begin{array}{l}{g_{z_{t}}^{(m)} \leftarrow \text { Attention }\left(\mathbf{Q}=g_{z_{t}}^{(m-1)}, \mathbf{K} \mathbf{V}=\mathbf{h}_{\mathbf{z}<t}^{(m-1)} ; \theta\right)} \\ {h_{z_{t}}^{(m)} \leftarrow \text { Attention }\left(\mathbf{Q}=h_{z_{t}}^{(m-1)}, \mathrm{KV}=\mathbf{h}_{\mathbf{z} \leq t}^{(m-1)} ; \theta\right)}\end{array}
$$

其中，$g_{i}^{(0)}=w, h_{i}^{(0)}=e\left(x_{i}\right)$ ，分别是随即初始化的向量和词向量。

t时刻的概率分布如下：

$$
p_{\theta}\left(X_{z_{t}}=x | \mathbf{x}_{z_{<t}}\right)=\frac{\exp \left(e(x)^{\top} g_{\theta}\left(\mathbf{x}_{\mathbf{z}<t} z_{t}\right)\right)}{\sum_{x^{\prime}} \exp \left(e\left(x^{\prime}\right)^{\top} g_{\theta}\left(\mathbf{x}_{\mathbf{z}<t}, z_{t}\right)\right)}
$$

其中，$z_t$表示的是位置向量，作用是：当词的位置不同，但是上文一样时，两个词算出来的概率是一样的。例如2-3-1-4-5和2-3-1-5-4，两个排列中，4和5的上文一样，算概率的时候就会一样【如下公式】，很显然这是错误的，不同位置的概率分布在ground-truth里是不一样的。

![](img/fun1.png)

引入位置向量之后，最终在预训练的时候也没有每一个token都预测，作者设置了一个超参数K，设定只预测序列最后1/K=的词[c+1, |z|]：

![](img/fun2.png)

### Transformer-XL

确定好目标函数之后，框架确定为Transformer-XL自回归语言模型。特点是relative positional encoding scheme和segment recurrence mechanism，更好地处理长文本，提升计算效率。具体不介绍了，详见参考文献[2]。


### Pretraining and Implementation

XLNet-Large和BERT-Large的参数量是差不多的，24层的Transformer-XL。经过处理，最终得到Wikipedia-2.78B，BooksCorpus-1.09B，Giga5-4.75B，ClueWeb-4.30B和Common Crawl respectively-19.97B，一共32.89B，近3倍于BERT的语料作为模型输入。序列长度512，memory维度384，batch-size为2048，用512块TPU v3跑了500K step用了2.5天。一般我种子才会设置成2048。

XLNet-Base和BERT-Base用的语料一样。但是貌似没说参数量一样。更详细的参数设置见论文补充材料[1]A.3。

## Conclusion

首先文章动机很明确，指出了当前AR LM和AE LM的缺点，在Transformer-XL的基础上结合AE捕获上下文的有点。

1. 提出了PLM预训练，用mask attention的方法实现factorization order的输入，大概率得到上下文信息。

2. 用two-stream self-attention，弥补自回归语言模型中目标预测歧义的缺陷。

本文实现factorization order的方式很巧妙，保证了pretrain/finetune的一致性。XLNet刷榜各大自然语言理解数据集，特别是RACE、SQuAD和RTE，但是用的计算资源真是令人叹为观止，门槛4块V100。

## Reference

[1]. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov and Quoc V. Le. [XLNet: Generalized Autoregressive Pretraining for Language Understanding.](https://arxiv.org/abs/1906.08237.pdf) arXiv preprint arXiv:1906.08237, 2019.

[2] Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. [Transformer-xl: Attentive language models beyond a ﬁxed-length context.](https://arxiv.org/abs/1901.02860.pdf) arXiv preprint arXiv:1901.02860, 2019.

[3] [论文笔记 — XLNet Generalized Autoregressive Pretraining for Language Understanding](https://indexfziq.github.io/2019/06/21/XLNet/)

[4] [XLNet Generalized Autoregressive Pretraining for Language Understanding 翻译](https://yuanxiaosc.github.io/2019/07/03/XLNet_Generalized_Autoregressive_Pretraining_for_Language_Understanding翻译/)




