# 【关于 P-tuning】 那些你不知道的事

> 作者：杨夕
> 
> 论文：GPT Understands, Too
> 
> 论文地址：https://arxiv.org/pdf/2104.08691.pdf
> 
> github: https://github.com/THUDM/P-tuning
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> NLP 面经地址：https://github.com/km1994/NLP-Interview-Notes
> 
> 推荐系统 百面百搭：https://github.com/km1994/RES-Interview-Notes
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 一、摘要

While GPTs with traditional fine-tuning fail to achieve strong results on natural language understanding (NLU), we show that GPTs can be better than or comparable to similar-sized BERTs on NLU tasks with a novel method P-tuning— which employs trainable continuous prompt embeddings. On the knowledge probing (LAMA) benchmark, the best GPT recovers 64% (P@1) of world knowledge without any additional text provided during test time, which substantially improves the previous best by 20+ percentage points. On the SuperGlue benchmark, GPTs achieve comparable and sometimes better performance to similar-sized BERTs in supervised learning. Importantly, we find that P-tuning also improves BERTs’ performance in both few-shot and supervised settings while largely reducing the need for prompt engineering. Consequently, Ptuning outperforms the state-of-the-art approaches on the few-shot SuperGlue benchmark

- 动机：虽然具有 Fine-tuning 的GPT在自然语言理解（NLU）方面无法取得很好的结果
- 论文方法：通过一种新的方法P-tuning，GPT在NLU任务上可以优于或相当于类似大小的BERT，该方法使用 trainable continuous prompt embeddings。
- 实验结果：在知识探测（LAMA）基准测试中，最佳GPT恢复率为64%(P@1)在测试期间没有提供任何额外文本的情况下，世界知识的水平大大提高了20多个百分点。在SuperGlue基准测试中，GPT在监督学习中实现了与类似大小的BERT相当的性能，有时甚至更好。重要的是，我们发现P调谐还提高了BERT在少镜头和监督设置中的性能，同时大大减少了对快速工程的需求。因此，在少镜头SuperGlue基准测试中，Ptuning优于最先进的方法

## 二、动机

1. 在 NLU任务上，GPT系列AR建模与 BERT双向语言模型相比有明显差距;
2. 研究表明：采用 prompt 训练方式 对 GPT3 进行 Fine-tuning， Fine-tuning 后的 GPT3 在 few-shot 和 zero-shot 上效果提升明显；
3. 传统的 prompt方法 存在明显缺陷：
   1. 人工制定prompt模板, 需要大量的验证集校验;
   2. 神经网络本身就是连续型建模，离散的prompts始终都只能是局部最优解

## 三、 P-tuning 方法

![](img/20230427220650.png)
> 一个 Prompt search “The capital of Britain is [MASK]””的例子。给定上下文（蓝色区域，“Britain”）和目标（红色区域，“[MASK]”），橙色区域指的是  prompt token 。在（a）中，prompt generato 仅接收离散奖励；相反，在（b）中，the pseudo prompts and prompt encode 可以以可微的方式进行优化。有时，添加少量与任务相关的锚定 tokens （如（b）中的“capital”）会带来进一步的改进

## 四、 P-tuning 论文实验



## 四、P-tuning 总结



## 参考

1. [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
2. [Tuning系列论文笔记](https://zhuanlan.zhihu.com/p/600119509)
3. [Prompt-Tuning、Instruction-Tuning和Chain-of-Thought](https://zhuanlan.zhihu.com/p/621480864)
4. [预训练新范式提示学习（Prompt-tuning，Prefix-tuning，P-tuning，PPT，SPoT）](https://blog.csdn.net/qq_39388410/article/details/121036309)
5. [关于大模型实践的一些总结](https://juejin.cn/post/7214318587429961786)





