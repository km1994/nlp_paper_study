# A Little Annotation does a Lot of Good A Study in Bootstrapping Low-resource Named Entity Recognizers
(一点注释对引导低资源命名实体识别器有很多好处)

## Abstract

大多数用于命名实体识别（NER）的最新模型都依赖于大量标记数据的可用性，这使它们很难扩展到资源较少的新语言。

但是，现在有几种提议的方法，涉及跨语言的转移学习（从其他资源丰富的语言中学习）或主动学习，该方法基于模型预测有效地选择有效的训练数据。

本文提出了一个问题：鉴于最近的进展以及有限的人工注释，以资源不足的语言有效创建高质量实体识别器的最有效方法是什么？基于使用模拟和真实人类注释的广泛实验，我们找到了最好的双重策略方法，首先是跨语言转移模型，然后仅在目标语言中对不确定的实体范围执行目标注释，从而最大程度地减少注释者的工作量。

结果表明，当可以注释很少的数据时，跨语言传输是一种强大的工具，但是以实体为目标的注释策略只需训练数据的十分之一即可快速达到竞争准确性。该代码可在此处公开获得。1

Most state-of-the-art models for named entity recognition (NER) rely on the availability of large amounts of labeled data, making them challenging to extend to new, lower resourced languages. However, there are now several proposed approaches involving either cross-lingual transfer learning, which learns from other highly resourced languages or active learning, which efficiently selects effective training data based on model predictions. This paper poses the question: given this recent progress, and limited human annotation, what is the most effective method for efficiently creating high-quality entity recognizers in under-resourced languages? Based on extensive experimentation using both simulated and real human annotation, we find a dual strategy approach best, starting with a cross-lingual transferred model, then performing targeted annotation of only uncertain entity spans in the target language, minimizing annotator effort. Results demonstrate that cross-lingual transfer is a powerful tool when very little data can be annotated, but an entity-targeted annotation strategy can achieve competitive accuracy quickly, with just one-tenth of training data. The code is publicly available here.

## 动机

有监督学习模型的效果依赖于语料的质量。

However, the performance of these models is highly dependent on the availability of large amounts of annotated data, and as a result, their accuracy is significantly lower on languages that have fewer resources than English. 


## 方法

在这项工作中，我们提出了一个问题：“我们如何仅需少量的人工就可以有效地引导低资源语言的高质量命名实体识别器？”

In this work, we ask the question “how can we efficiently bootstrap a high-quality named entity recognizer for a low-resource language with only a small amount of human effort?” 

特别地，我们利用数据有效学习的最新进展来降低资源语言，提出以下“recipe” 以引导低资源实体识别器：首先，我们使用跨语言迁移学习（Yarowsky等人，2001； Ammar等人，2016），该模型应用了在另一种语言上训练的模型语言到资源匮乏的语言，为启动引导过程提供了良好的初步模型。具体来说，我们使用谢等人的模型。 （2018），报告了许多语言对的出色结果。接下来，在此转移模型的基础上，我们进一步采用主动学习（Settles和Craven，2008； Marchegigiani和Artieres，2014），这通过使用模型预测为人类注释者选择信息性数据而非随机数据来帮助提高注释效率。最后，根据使用主动学习获得的数据对模型进行微调，以提高目标语言的准确性。

Specifically, we leverage recent advances in data-efficient learning for low-resource languages, proposing the following “recipe” for bootstrapping low-resource entity recognizers: First, we use cross-lingual transfer learning (Yarowsky et al., 2001; Ammar et al., 2016), which applies a model trained on another language to low-resource languages, to provide a good preliminary model to start the bootstrapping process. Specifically, we use the model of Xie et al. (2018), which reports strong results on a number of language pairs. Next, on top of this transferred model, we further employ active learning (Settles and Craven, 2008; Marcheggiani and Artieres, 2014), which helps improve annotation efficiency by using model predictions to select informative, rather than random, data for human annotators. Finally, the model is fine-tuned on data obtained using active learning to improve accuracy in the target language. 

在此配方中，选择一种用于在主动学习中选择和注释数据的特定方法对于最大限度地减少人工工作非常重要。在NER的先前工作中使用的一种相对标准的方法是，根据识别其中的实体的不确定性的标准来选择完整序列（Culotta和McCallum，2005年）。但是，由于通常只有句子中的单个实体可能是令人感兴趣的情况，因此当仅对句子中的一小部分感兴趣时，注释完整的序列仍然很繁琐和浪费（Neubig等，2011 ; Sperber et al。，2014）。受此发现启发，并考虑到命名实体既重要又稀疏的事实，我们提出了一种以实体为目标的策略，以节省注释者的工作量。具体来说，我们选择序列中最可能命名的实体的不确定的子代跨度。这样，注释者只需要将类型分配给选定的子范围，而无需读取和注释整个序列。为了应对由此产生的部分序列注释，我们在训练过程中应用了条件随机字段（CRF）的约束版本，即部分CRF，这些条件只能从带注释的子跨度中学习（Tsuboi等人，2008; Wanvarie等人，2011 ）。

Within this recipe, the choice of a specific method for choosing and annotating data within active learning is highly important to minimize human effort. One relatively standard method used in previous work on NER is to select full sequences based on a criterion for the uncertainty of the entities recognized therein (Culotta and McCallum, 2005). However, as it is often the case that only a single entity within the sentence may be of interest, it can still be tedious and wasteful to annotate full sequences when only a small portion of the sentence is of interest (Neubig et al., 2011; Sperber et al., 2014). Inspired by this finding and considering the fact that named entities are both important and sparse, we propose an entity-targeted strategy to save annotator effort. Specifically, we select uncertain sub spans of tokens within a sequence that are most likely named entities. This way, the annotators only need to assign types to the chosen sub spans without having to read and annotate the full sequence. To cope with the resulting partial annotation of sequences, we apply a constrained version of conditional random fields (CRFs), partial CRFs, during training that only learn from the annotated sub spans (Tsuboi et al., 2008; Wanvarie et al., 2011).





