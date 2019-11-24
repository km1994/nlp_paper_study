# FreeLB: Enhanced Adversarial Training for Language Understanding 加强语言理解的对抗性训练

## Abstract

Adversarial training, which minimizes the maximal risk for label-preserving input perturbations, has proved to be effective for improving the generalization of language models. In this work, we propose a novel adversarial training algorithm - FreeLB, that promotes higher robustness and invariance in the embedding space, by adding adversarial perturbations to word embeddings and minimizing the resultant adversarial risk inside different regions around input samples. To validate the effectiveness of the proposed approach, we apply it to Transformer-based models for natural language understanding and commonsense reasoning tasks. Experiments on the GLUE benchmark show that when applied only to the finetuning stage, it is able to improve the overall test scores of BERT-based model from 78.3 to 79.4, and RoBERTa-large model from 88.5 to 88.8. In addition, the proposed approach achieves state-of-the-art test accuracies of 85.39% and 67.32% on ARC-Easy and ARC-Challenge. Experiments on CommonsenseQA benchmark further demonstrate that FreeLB can be generalized and boost the performance of RoBERTa-large model on other tasks as well.

对抗训练使保留标签的输入扰动的最大风险最小，对于提高语言模型的泛化能力是有效的。 

在这项工作中，我们提出了一种新的对抗性训练算法—— freeb，它通过在字嵌入中添加对抗性的干扰，最小化输入样本周围不同区域内的对抗性风险，从而提高嵌入空间的鲁棒性和不变性。 

为了验证该方法的有效性，我们将其应用到基于 transformer 的自然语言理解模型和常识推理任务中。 Glue 基准测试的实验结果表明，当仅应用于细化阶段时，它能够将基于 bert 模型的总体测试分数从78.3提高到79.4，并将 RoBERTa-large 模型的总体测试分数从88.5提高到88.8。 此外，该方法在 arceasy 和 ARC-Challenge 上分别实现了85.39% 和67.32% 的最新测试准确率。 Commonsenseqa 基准测试的实验进一步证明，freeb 可以被广义化，并且可以提高 RoBERTa-large 模型在其他任务上的性能。

## 对抗训练介绍

对抗性训练是一种创建健壮神经网络的方法。 在对抗性训练期间，小批量的训练样本受到对抗性扰动的污染(这些扰动很小，但会导致错误分类) ，然后用于更新网络参数，直到最终的模型学会抵抗这种攻击。 对抗性训练最初是作为提高机器学习系统(43405)安全性的一种手段而提出的，特别是对于安全关键系统，如自动驾驶汽车(Xiao 2018 eccv)和版权检测(saadatpanah2019 adversal)。

## 方法提出

本文展示对抗性训练对于许多语言理解任务来说，显著地提高了最先进模型的性能。 特别是，提出了一种新的对抗性训练算法，称为 freeb (Free Large-Batch) ，它给单词嵌入增加了对抗性干扰，并将输入样本周围的对抗性损失降到最低。 该方法利用最近提出的“full” 训练策略 ，在不同的标准约束下使用多样化的对抗性样本来扩大批量，而且不会比基于 pgd (预测梯度下降法)的对抗性培训(madry2018towards)增加额外的成本，这使我们能够在大规模的最先进的模型上进行这种多样化的对抗性培训。 我们观察到用 freeb 训练的模型在嵌入空间中的鲁棒性和不变性得到了改善，并且与泛化成正相关。

## ADVERSARIAL TRAINING FOR LANGUAGE UNDERSTANDING

### PGD FOR ADVERSARIAL TRAINING   PGD 对抗体训练中心

标准对抗训练的目的是找出一系列参数 $\theta^{*}$ ，以尽量减少正常球内任何一个$\delta$的最大风险:

![1](img/1.png)

其中 D 是数据分布，y 是标号，L 是某个损失函数。 我们使用 Frobenius 范数来约束 $\delta$。 对于神经网络，外部的“最小”是非凸的，内部的“最大”是非凹的。 然而，2018年的马德里论证表明，这个鞍点问题可以可靠地解决与 SGD 的外部最小化和 PGD (大规模约束优化的标准方法，见 combettes2011近端和 goldstein2014场)的内部最大化。

特别地，对于约束 ${||δ||}_{F}≤ϵ$ ，在附加损失函数为局部线性的假设下，PGD 在每次迭代中采取以下步骤(步长) :

![](img/2.png)

### LARGE-BATCH ADVERSARIAL TRAINING FOR FREE

![](img/3.png)

## 结论

开发了一种对抗性的训练方法 freeb，以提高自然语言的理解能力。 该方法采用梯度方法对连续的词嵌入进行干扰，有效地最小化了结果的对抗性风险。 Freelb 能够在多个数据集上增强基于 transformer 的模型(BERT 和 RoBERTa) ，并在 GLUE 和 ARC 基准测试上达到新的水平。 