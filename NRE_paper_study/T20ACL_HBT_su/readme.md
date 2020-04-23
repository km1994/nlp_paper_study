# A Novel Hierarchical Binary Tagging Framework for Relational Triple Extraction

## 摘要

Extracting relational triples from unstructured text is crucial for large-scale knowledge graph construction. However, few existing works excel in solving the overlapping triple problem where multiple relational triples in the same sentence share the same entities. In this work, we introduce a fresh perspective to revisit the relational triple extraction task and propose a novel Hierarchical Binary Tagging (HBT) framework derived from a principled problem formulation. Instead of treating relations as discrete labels as in previous works, our new framework models relations as functions that map subjects to objects in a sentence, which naturally handles the overlapping problem. Experiments show that the proposed framework already outperforms state-of-the-art methods even when its encoder module uses a randomly initialized BERT encoder, showing the power of the new tagging framework. It enjoys further performance boost when employing a pretrained BERT encoder, outperforming the strongest baseline by 17.5 and 30.2 absolute gain in F1-score on two public datasets NYT and WebNLG, respectively. In-depth analysis on different scenarios of overlapping triples shows that the method delivers consistent performance gain across all these scenarios.

从非结构化文本中提取关系三元组对于大规模知识图的构建至关重要。

但是，很少有现有的著作能很好地解决重叠三重问题，在该问题中，同一句子中的多个关系三重共享同一实体。

在这项工作中，我们引入了一个新的视角来重新审视关系三重提取任务，并提出了一个从有原则的问题表达中衍生出来的新颖的分层二进制标记（HBT）框架。我们的新框架没有像以前的作品那样将关系视为离散标签，而是将关系建模为将主语映射到句子中的宾语的函数，从而自然地解决了重叠问题。

实验表明，即使在其编码器模块使用随机初始化的BERT编码器的情况下，所提出的框架也已经超越了最新方法，显示了新标签框架的强大功能。当使用预训练的BERT编码器时，它在性能上得到了进一步的提升，在两个公共数据集NYT和WebNLG上，F1评分的绝对增益分别比最强的基线高17.5和30.2。对重叠三元组的不同方案的深入分析表明，该方法在所有这些方案中均提供了一致的性能提升。

## 引言

### 之前方法介绍

#### pipeline approach

##### 思路

1. 识别句子中的所有实体，
2. 对每个实体对执行关系分类。 
   
##### 问题

由于早期阶段的错误无法在后期阶段进行纠正，因此这种方法容易遭受错误传播问题的困扰。

#### feature-based models and neural network-based models 

##### 思路

通过用学习表示替换人工构建的特征，基于神经网络的模型在三重提取任务中取得了相当大的成功。

##### 问题

大多数现有方法无法正确处理句子包含多个相互重叠的关系三元组的情况。

<img src="img/1.png" width=700>

> 图1说明了这些场景，其中三元组在一个句子中共享一个或两个实体。 这个重叠的三重问题直接挑战了传统的序列标签方案，该方案假定每个令牌仅带有一个标签。 假设一个实体对最多拥有一个关系的关系分类方法也给它带来了很大的困难。

#### 基于Seq2Seq模型  and GCN

##### 思路

Zeng是最早在关系三重提取中考虑重叠三重问题的人之一。 他们介绍了如图1所示的不同重叠模式的类别，并提出了具有复制机制以提取三元组的序列到序列（Seq2Seq）模型。 他们基于Seq2Seq模型，进一步研究了提取顺序的影响，并通过强化学习获得了很大的改进。 

Fu还通过使用基于图卷积网络（GCN）的模型将文本建模为关系图来研究重叠三重问题。

##### 问题

1. 它们都将关系视为要分配给实体对的离散标签。 这种表述使关系分类成为硬机器学习问题。 首先，班级分布高度不平衡。 在所有提取的实体对中，大多数都不形成有效关系，从而产生了太多的否定实例。 其次，当同一实体参与多个有效关系（重叠三元组）时，分类器可能会感到困惑。 没有足够的训练示例，分类器就很难说出实体参与的关系。结果，提取的三元组通常是不完整且不准确的。


## 本文工作

在这项工作中，我们从 triple 的关系提取的原则性公式开始。 这产生了一种通用的算法框架，该框架通过设计来处理重叠的 triple 问题。 

该框架的核心是崭新的视角，我们可以将关系建模为将主体映射到对象的函数，而不是将关系视为实体对上的离散标签。 

更准确地说，我们不是学习关系分类器f（s，o）→r，而是学习关系特定的标记器fr（s）→o，

每个标记器都可以识别特定关系下给定 subject 的可能 object(s)。 或不返回任何 object，表示给定的主题和关系没有 triple。 

在这种框架下，triple 提取是一个分为两个步骤的过程：首先，我们确定句子中所有可能的 subjects； 然后针对每个subjects，我们应用特定于关系的标记器来同时识别所有可能的 relations 和相应的 objects。

We implement the above idea in an end-to-end hierarchical binary tagging (HBT) framework. It consists of a BERT-based encoder module, a subject tagging module, and a relation-specific object tagging module. 

## 贡献

1. We introduce a fresh perspective to revisit the relational triple extraction task with a principled problem formulation, which implies a general algorithmic framework that addresses the overlapping triple problem by design. 

2. We instantiate the above framework as a novel hierarchical binary tagging model on top of a Transformer encoder. This allows the model to combine the power of the novel tagging framework with the prior knowledge in pretrained large-scale language models. 

3. Extensive experiments on two public datasets show that the proposed framework overwhelmingly outperforms state-of-the-art methods, achieving 17.5 and 30.2 absolute gain in F1-score on the two datasets respectively. Detailed analyses show that our model gains consistent improvement in all scenarios. 

## 结论

In this paper, we introduce a novel hierarchical binary tagging (HBT) framework derived from a principled problem formulation for relational triple extraction. Instead of modeling relations as discrete labels of entity pairs, we model the relations as functions that map subjects to objects, which provides a fresh perspective to revisit the relational triple extraction task. As a consequent, our model can simultaneously extract multiple relational triples from sentences, without suffering from the overlapping problem. We conduct extensive experiments on two widely used datasets to validate the effectiveness of the proposed HBT framework. Experimental results show that our model overwhelmingly outperforms state-of-theart baselines over different scenarios, especially on the extraction of overlapping relational triples.

在本文中，我们介绍了一种新颖的层次化二进制标记（HBT）框架，该框架源自用于关系三重提取的原则性问题公式。 我们不是将关系建模为实体对的离散标签，而是将关系建模为将主题映射到对象的函数，这为重新审视关系三重提取任务提供了新的视角。 因此，我们的模型可以同时从句子中提取多个关系三元组，而不会出现重叠问题。 我们对两个广泛使用的数据集进行了广泛的实验，以验证所提出的HBT框架的有效性。 实验结果表明，在不同情况下，特别是在提取重叠的关系三元组时，我们的模型绝对优于最新的基准。