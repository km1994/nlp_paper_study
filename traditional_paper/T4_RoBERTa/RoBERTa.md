# RoBERTa: A Robustly Optimized BERT Pretraining Approach

## 摘要

Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, hyperparameter choices have significant impact on the final results. We present a replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the importance of previously overlooked design choices, and raise questions about the source of recently reported improvements. We release our models and code.

语言模型的预训练已导致显着的性能提升，但是不同方法之间的仔细比较是具有挑战性的。 训练在计算上很昂贵，通常是在不同大小的私人数据集上进行的，而且正如我们将显示的那样，超参数的选择对最终结果有重大影响。 我们提出了BERT预训练的重复研究（Devlin等人，2019），该研究仔细衡量了许多关键超参数和训练数据量的影响。 我们发现BERT的训练不足，并且可以匹配或超过其发布的每个模型的性能。 我们最好的模型在GLUE，RACE和SQuAD上获得了最先进的结果。 这些结果突出了以前被忽略的设计选择的重要性，并引起了人们对最近报告的改进来源的质疑。 我们发布我们的模型和代码。


## 动机

- Bert 序列模型的问题
> 确定方法的哪些方面贡献最大可能是具有挑战性的;
> 
> 训练在计算上是昂贵的的，限制了可能完成的调整量


## 所做工作概述

We present a replication study of BERT pretraining (Devlin et al., 2019), which includes a careful evaluation of the effects of hyperparmeter tuning and training set size. We find that BERT was significantly undertrained and propose an improved recipe for training BERT models, which we call RoBERTa, that can match or exceed the performance of all of the post-BERT methods. Our modifications are simple, they include: (1) training the model longer, with bigger batches, over more data; (2) removing the next sentence prediction objective; (3) training on longer sequences; and (4) dynamically changing the masking pattern applied to the training data. We also collect a large new dataset (CC-NEWS) of comparable size to other privately used datasets, to better control for training set size effects.

我们提出了BERT预训练的复制研究（Devlin等人，2019），其中包括对超参数调整和训练集大小的影响的仔细评估。 我们发现BERT的训练不足，并提出了一种改进的方法来训练BERT模型（我们称为RoBERTa），该模型可以匹配或超过所有后BERT方法的性能。 我们的修改很简单，其中包括：

- 更大的模型参数量（论文提供的训练时间来看，模型使用 1024 块 V100 GPU 训练了 1 天的时间）
- 更大bacth size。RoBERTa 在训练过程中使用了更大的bacth size。尝试过从 256 到 8000 不等的bacth size。
- 更多的训练数据（包括：CC-NEWS 等在内的 160GB 纯文本。而最初的BERT使用16GB BookCorpus数据集和英语维基百科进行训练）
另外，RoBERTa在训练方法上有以下改进：

- 去掉下一句预测(NSP)任务
- 动态掩码。BERT 依赖随机掩码和预测 token。原版的 BERT 实现在数据预处理期间执行一次掩码，得到一个静态掩码。 而 RoBERTa 使用了动态掩码：每次向模型输入一个序列时都会生成新的掩码模式。这样，在大量数据不断输入的过程中，模型会逐渐适应不同的掩码策略，学习不同的语言表征。
- 文本编码。Byte-Pair Encoding（BPE）是字符级和词级别表征的混合，支持处理自然语言语料库中的众多常见词汇。原版的 BERT 实现使用字符级别的 BPE 词汇，大小为 30K，是在利用启发式分词规则对输入进行预处理之后学得的。Facebook 研究者没有采用这种方式，而是考虑用更大的 byte 级别 BPE 词汇表来训练 BERT，这一词汇表包含 50K 的 subword 单元，且没有对输入作任何额外的预处理或分词。


## 贡献

In summary, the contributions of this paper are: (1) We present a set of important BERT design choices and training strategies and introduce alternatives that lead to better downstream task performance; (2) We use a novel dataset, CCNEWS, and confirm that using more data for pretraining further improves performance on downstream tasks; (3) Our training improvements show that masked language model pretraining, under the right design choices, is competitive with all other recently published methods. We release our model, pretraining and fine-tuning code implemented in PyTorch (Paszke et al., 2017).

总而言之，本文的贡献是：（1）我们提出了一组重要的BERT设计选择和训练策略，并介绍了可以改善下游任务性能的替代方法； （2）我们使用新的数据集CCNEWS，并确认使用更多数据进行预训练可进一步提高下游任务的性能； （3）我们的训练改进表明，在正确的设计选择下，屏蔽语言模型预训练与所有其他最近发布的方法相比具有竞争力。 我们发布了在PyTorch中实现的模型，预训练和微调代码（Paszke et al。，2017）。


## 实现方式

We reimplement BERT in FAIRSEQ (Ott et al., 2019). We primarily follow the original BERT optimization hyperparameters, given in Section 2, except for the peak learning rate and number of warmup steps, which are tuned separately for each setting. We additionally found training to be very sensitive to the Adam epsilon term, and in some cases we obtained better performance or improved stability after tuning it. Similarly, we found setting β2 = 0.98 to improve stability when training with large batch sizes. We pretrain with sequences of at most T = 512 tokens. Unlike Devlin et al. (2019), we do not randomly inject short sequences, and we do not train with a reduced sequence length for the first 90% of updates. We train only with full-length sequences. We train with mixed precision floating point arithmetic on DGX-1 machines, each with 8 × 32GB Nvidia V100 GPUs interconnected by In- finiband (Micikevicius et al., 2018).

我们在FAIRSEQ中重新实现BERT（Ott等，2019）。 我们主要遵循第2节中给出的原始BERT优化超参数，除了峰值学习速率和预热步骤数（针对每个设置分别调整）之外。 我们还发现训练对 Adam epsilon term 非常敏感，并且在某些情况下，对其进行调整后可以获得更好的性能或更高的稳定性。 

同样，我们发现设置β2= 0.98可以提高大批量训练时的稳定性。 我们使用最多T = 512个令牌的序列进行预训练。 与Devlin等人不同。 （2019），我们不会随机注入短序列，并且对于前90％的更新，我们不会以缩短的序列长度进行训练。 我们只训练全长序列。 我们在DGX-1机器上使用混合精度浮点算术进行训练，每台机器都配有由Infiniband互连的8×32GB Nvidia V100 GPU（Micikevicius等，2018）。


## 结论

We carefully evaluate a number of design decisions when pretraining BERT models. We find that performance can be substantially improved by training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. Our improved pretraining procedure, which we call RoBERTa, achieves state-of-the-art results on GLUE, RACE and SQuAD, without multi-task finetuning for GLUE or additional data for SQuAD. These results illustrate the importance of these previously overlooked design decisions and suggest that BERT’s pretraining objective remains competitive with recently proposed alternatives. We additionally use a novel dataset, CC-NEWS, and release our models and code for pretraining and finetuning at: https://github.com/pytorch/fairseq.

在预训练BERT模型时，我们会仔细评估许多设计决策。 我们发现，通过对模型进行较长时间的训练（使用更多批次的数据处理更多数据），可以显着提高性能。 删除下一个句子预测目标； 训练更长的序列； 并动态更改应用于训练数据的掩蔽模式。 我们改进的预训练程序（称为RoBERTa）可在GLUE，RACE和SQuAD上获得最新的结果，而无需为GLUE进行多任务微调或为SQuAD进行其他数据调整。 这些结果说明了这些先前被忽略的设计决策的重要性，并表明BERT的预训练目标与最近提出的替代方案相比仍然具有竞争力。 我们还使用一个新颖的数据集CC-NEWS，并在以下网址发布我们的模型和代码以进行预训练和微调：https://github.com/pytorch/fairseq。


## 参考文献

1. [文献阅读笔记:RoBERTa：A Robustly Optimized BERT Pretraining Approach](https://blog.csdn.net/ljp1919/article/details/100666563)