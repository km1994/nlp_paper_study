# TENER: Adapting Transformer Encoder for Name Entity Recognition

## 摘要

BILSTM 被广泛应用于命名实体识别任务上；

动机：Transformer 在 NER 效果问题；

方法：TENER：adapted Transformer Encoder to model the character-level features and word-level features

The Bidirectional long short-term memory networks (BiLSTM) have been widely used as an encoder in models solving the named entity recognition (NER) task. 

Recently, the Transformer is broadly adopted in various Natural Language Processing (NLP) tasks owing to its parallelism and advantageous performance. Nevertheless, the performance of the Transformer in NER is not as good as it is in other NLP tasks. 

In this paper, we propose TENER, a NER architecture adopting adapted Transformer Encoder to model the character-level features and word-level features. 

By incorporating the direction and relative distance aware attention and the un-scaled attention, we prove the Transformer-like encoder is just as effective for NER as other NLP tasks. Experiments on six NER datasets show that TENER achieves superior performance than the prevailing BiLSTM-based models.


## 动机

1. Transformer 能够解决长距离依赖问题；
2. Transformer 能够并行化；
3. 然而，Transformer 在 NER 任务上面效果不好。

 The Transformer encoder adopts a fully-connected self-attention structure to model the long-range context, which is the weakness of RNNs. Moreover, Transformer has better parallelism ability than RNNs. 
 
 However, in the NER task, Transformer encoder has been reported to perform poorly (Guo et al., 2019), our experiments also confirm this result. Therefore, it is intriguing to explore the reason why Transformer does not work well in NER task.

 ## 论文创新点

1. 引入：相对位置编码；
2. 

The first is that the sinusoidal position embedding used in the vanilla Transformer is relative distance sensitive and direction-agnostic, but this property will lose when used in the vanilla Transformer. However, both the direction and relative distance information are important in the NER task. For example, words after “in” are more likely to be a location or time than words before it, and words before “Inc.” are mostly likely to be of the entity type “ORG”. Besides, an entity is a continuous span of words. Therefore, the awareness of relative distance might help the word better recognizes its neighbor. To endow the Transformer with the ability of directionality and relative distance awareness, we adopt a direction-aware attention with the relative positional encoding (Shaw et al., 2018; Huang et al., 2019; Dai et al., 2019). We propose a revised relative positional encoding that uses fewer parameters and performs better.

![](img/1.png)


The second is an empirical finding. The attention distribution of the vanilla Transformer is scaled and smooth. But for NER, a sparse attention is suitable since not all words are necessary to be attended. Given a current word, a few contextual words are enough to judge its label. The smooth attention could include some noisy information. Therefore, we abandon the scale factor of dot-production attention and use an un-scaled and sharp attention.（第二是经验发现。 香草变压器的注意力分布是缩放且平滑的。 但是对于NER，因为并非所有单词都需要参加，所以很少注意是合适的。 给定一个当前单词，只需几个上下文单词就足以判断其标签。 平稳的注意力可能包括一些嘈杂的信息。 因此，我们放弃了点生产注意力的比例因子，而使用了无比例且敏锐的注意力。）
