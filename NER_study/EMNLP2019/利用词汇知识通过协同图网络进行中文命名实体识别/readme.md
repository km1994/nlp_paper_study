# Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network
(利用词汇知识通过协同图网络进行中文命名实体识别)

## Abstract

单词边界信息的缺乏已被视为开发高性能中文命名实体识别（NER）系统的主要障碍之一。 幸运的是，自动构建的词典包含丰富的单词边界信息和单词语义信息。 然而，将词汇知识整合到中文NER任务中时，在自匹配词汇词以及最接近的上下文词汇词方面仍然面临挑战。 

我们提出了一个协作图网络来解决这些挑战。 在各种数据集上进行的实验表明，我们的模型不仅超越了最新技术（SOTA）的结果，而且实现了比SOTA模型快6到15倍的速度。1

The lack of word boundaries information has been seen as one of the main obstacles to develop a high performance Chinese named entity recognition (NER) system. Fortunately, the automatically constructed lexicon contains rich word boundaries information and word semantic information. However, integrating lexical knowledge in Chinese NER tasks still faces challenges when it comes to self-matched lexical words as well as the nearest contextual lexical words. We present a Collaborative Graph Network to solve these challenges. Experiments on various datasets show that our model not only outperforms the stateof-the-art (SOTA) results, but also achieves a speed that is six to fifteen times faster than that of the SOTA model.1 

## 研究现状

将词信息纳入NER的主要方法有三种。

- 第一个是管道方法。流水线方法的方法是首先应用中文分词（CWS），然后使用基于单词的NER模型。但是，由于CWS的错误可能会影响NER的性能，因此流水线方法会出现错误传播。

- 第二个是共同学习CWS和NER任务（Xu等人，2013; Peng和Dredze，2016; Cao等人，2018; Wu等人，2019）。但是，联合模型必须依赖CWS注释数据集，该数据集成本高昂并且在许多不同的分割标准下进行注释（Chen等人，2017）。

- 第三个是利用自动构造的词典，该词典在大型自动分段的文本上进行了预训练。词汇知识包括边界和语义信息。边界信息由词典词本身提供，语义信息由预训练的词嵌入提供（Bengio等，2003； Mikolov等，2013）。与联合方法相比，词典易于获得，并且不需要其他注释CWS数据集。最近，Zhang和Yang（2018）提出了一种lattice LSTM，将词法知识整合到NER中。然而，将词汇知识整合到句子中仍然面临两个挑战。

There are three main ways to incorporate word information in NER. The first one is the pipeline method. The way of pipeline method is to apply Chinese Word Segmentation (CWS) first, and then to use a word-based NER model. However, the pipeline method suffers from error propagation, since the error of CWS may affect the performance of NER. The second one is to learn CWS and NER tasks jointly (Xu et al., 2013; Peng and Dredze, 2016; Cao et al., 2018; Wu et al., 2019). However, the joint models must rely on CWS annotation datasets, which are costly and are annotated under many diverse segmentation criteria (Chen et al., 2017). The third one is to leverage an automatically constructed lexicon, which is pre-trained on large automatically segmented texts. Lexical knowledge includes boundaries and semantic information. Boundaries information is provided by the lexicon word itself, and semantic information is provided by pre-trained word embeddings (Bengio et al., 2003; Mikolov et al., 2013). Compared with joint methods, a lexicon is easy to obtain and additional annotation CWS datasets are not required. Recently, Zhang and Yang (2018) propose a lattice LSTM to integrate lexical knowledge in NER. However, integrating lexical knowledge into sentences still faces two challenges.

## 动机

词语的边界和命名实体的边界相似。

在本文中，我们重点介绍中文NER。 与英语相比，中文没有明显的单词边界。 由于没有单词边界信息，因此仅对中文NER使用字符信息是很直观的（He和Wang，2008； Liu等，2010； Li等，2014），尽管这种方法可能会导致忽略单词信息 。 但是，单词信息在中文NER中非常有用，因为单词边界通常与命名实体边界相同。 例如，如图1所示，单词“北京机场”（北京机场）的边界与命名实体“北京机场”（北京机场）的边界相同。 因此，充分利用单词信息将有助于提高中文NER成绩。

![](img/1.png)

In this paper, we focus on Chinese NER. Compared with English, Chinese has no obvious word boundaries. Since without word boundaries information, it is intuitive to use character information only for Chinese NER (He and Wang, 2008; Liu et al., 2010; Li et al., 2014), although such methods could result in the disregard of word information. However, word information is very useful in Chinese NER, because word boundaries are usually the same as named entity boundaries. For example, as shown in Figure 1, the boundaries of the word “北京机场” (Beijing airport) are the same as the boundaries of the named entity “ 北京机场” (Beijing airport). Therefore, making full use of word information would help to improve the Chinese NER performance.

第一个挑战是整合自我匹配的词汇词。 字符的自匹配词汇词是包含此字符的词汇词。 例如，“北京机场”（北京机场）和“机场”（机场）是字符“机”（飞机）的自匹配单词。 “离开”（离开）不是字符“机”（飞机）的自匹配单词，因为单词“离开”（离开）中不包含“机”（飞机）。 自匹配词的词汇知识在中文NER中很有用。 例如，如图1所示，自匹配单词“北京机场”（Beijing Airport）的边界和语义知识可以帮助字符“机”（飞机）预测“ I-LOC”标签，而不是“ O” ”或“ B-LOC”标签。 然而，由于单词字符 lattice 的限制，lattice LSTM（Zhang和Yang，2018）未能将自匹配单词“北京机场”（北京机场）整合到字符“机”（飞机）中。

The first challenge is to integrate self-matched lexical words. A self-matched lexical word of a character is the lexical word that contains this character. For instance, “北京机场” (Beijing Airport) and “机场” (Airport) are the self-matched words of the character “机” (airplane). “离开” (leave) is not the self-matched word of the character “:” (airplane), since “ 机” (airplane) is not contained in the word “离开” (leave). The lexical knowledge of self-matched word is useful in Chinese NER. For example, as shown in Figure 1, the boundaries and semantic knowledge of the selfmatched word “北京机场” (Beijing Airport) can help the character “机”(airplane) to predict an “I-LOC” tag, instead of “O” or “B-LOC” tags. However, due to the limits of the word-character lattice, the lattice LSTM (Zhang and Yang, 2018) fails to integrate the self-matched word “北京机场” (Beijing Airport) into the character “机” (airplane).

第二个挑战是直接整合最接近的上下文词汇词。字符的最接近上下文词汇词是与该字符给定句子中最接近的过去或将来子序列匹配的词。例如，词汇词“离开”（leave）是字符“顿”（-ton）的最近上下文词，因为该词与字符的最近将来子序列“离开”相匹配，而“北京¨”（北京）不是该字符最接近的上下文词汇词。最接近的上下文词汇词对中文NER有利。例如，如图1所示，通过直接使用最近的上下文单词“离开”（leave）的语义知识，可以预测“ I-PER”标签而不是“ I-ORG”标签，因为“希尔顿”（希尔顿酒店）不能作为动词“离开”（离开）的主语。然而，格模型（Zhan and Yang，2018）仅通过先前的隐藏状态隐式地整合了最近的上下文词汇词的知识。最接近的上下文词汇词的信息可能会受到其他信息的干扰。

 The second challenge is to integrate the nearest contextual lexical words directly. The nearest contextual lexical word of a character is the word that matches the nearest past or future subsequence in the given sentence of this character. For instance, the lexical word “离开” (leave) is the nearest contextual word of the character “顿” (-ton), since the word matches the nearest future subsequence “离开” of the character, while “北京¨” (Beijing) is not the nearest contextual lexical word of this character. The nearest contextual lexical words are beneficial for Chinese NER. For example, as shown in Figure 1, by directly using the semantic knowledge of the nearest contextual words “离开” (leave), an “I-PER” tag can be predicted instead of an “I-ORG” tag, since “希尔顿” (Hilton Hotels) cannot be taken as the subject of the verb “离开 ” (leave). However, a lattice model (Zhan  and Yang, 2018) only implicitly integrate the knowledge of the nearest contextual lexical words via the previous hidden state. The information of the nearest contextual lexical word may be disturbed by other information.

## 方法

离子。为了解决上述挑战，我们提出了一种基于字符的协作图网络，包括编码层，图层，融合层和解码层。具体来说，在图层中有三个单词-字符交互图。第一个是包含图（C-graph），它用于集成自匹配词法单词。它模拟了字符和自匹配词汇词之间的联系。第二个是过渡图（T-graph），它建立了字符和最接近的上下文匹配单词之间的直接连接。它有助于应对直接集成最接近的上下文单词的挑战。第三个是 Lattice 图（L-graph），它受格子LSTM（Zhang and Yang，2018）的启发。 L-graph通过多次跳跃隐式捕获自匹配词汇词和最近的上下文词汇词的部分信息。这些图的构建没有外部NLP工具，可以避免错误传播问题。此外，这些图很好地互补，并且为这些图之间的协作设计了融合层。

To solve the above challenges, we propose a character-based Collaborative Graph Network, including an encoding layer, a graph layer, a fusion layer and a decoding layer. Specifically, there are three word-character interactive graphs in the graph layer. The first one is the Containing graph (C-graph), which is designed for integrating self-matched lexical words. It models the connection between characters and self-matched lexical words. The second one is the Transition graph (T-graph), which builds the direct connection between characters and the nearest contextual matched words. It helps to handle the challenge of integrating the nearest contextual words directly. The third one is the Lattice graph (L-graph), which is inspired by the lattice LSTM (Zhang and Yang, 2018). L-graph captures partial information of self-matched lexical words and the nearest contextual lexical words implicitly by multiple hops. These graphs are built without external NLP tools, which can avoid error propagation problem. Besides, these graphs complement each other nicely and a fusion layer is designed for collaboration between these graph.


## 结论

在本文中，我们提出了一个用于整合中文NER中词汇知识的协作图网络。 网络的核心是三个词汇词-字符交互图。 这些交互式图可以捕获不同的词汇知识，并且无需外部NLP工具即可构建。 通过各种实验，我们证明了我们的模型与SOTA模型具有互补的优势，并且这些交互图是有效的。

In this paper, we propose a Collaborative Graph Network for integrating lexical knowledge in Chinese NER. The core of the network is three lexical word-character interactive graphs. These interactive graphs can capture different lexical knowledge and are built without external NLP tools. We show through various experiments that our model has complementary strengths to the SOTA model and these interactive graphs are effective.

https://github.com/DianboWork/Graph4CNER.git