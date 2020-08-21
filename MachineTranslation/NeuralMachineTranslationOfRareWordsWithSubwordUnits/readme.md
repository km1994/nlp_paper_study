# Neural Machine Translation of Rare Words with Subword Units

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。


## 摘要

Neural machine translation (NMT) models typically operate with a fixed vocabulary, but translation is an open-vocabulary problem. Previous work addresses the translation of out-of-vocabulary words by backing off to a dictionary. In this paper, we introduce a simpler and more effective approach, making the NMT model capable of open-vocabulary translation by encoding rare and unknown words as sequences of subword units. This is based on the intuition that various word classes are translatable via smaller units than words, for instance names (via character copying or transliteration), compounds (via compositional translation), and cognates and loanwords (via phonological and morphological transformations). We discuss the suitability of different word segmentation techniques, including simple character ngram models and a segmentation based on the byte pair encoding compression algorithm, and empirically show that subword models improve over a back-off dictionary baseline for the WMT 15 translation tasks English→German and English→Russian by up to 1.1 and 1.3 BLEU, respectively.

- 介绍：神经机器翻译（NMT）模型通常以固定的词汇量运行，但是翻译是一个开放词汇的问题。
- 先前的工作：通过退回到字典来解决词汇外单词的翻译。
- 本文工作：介绍了一种更简单，更有效的方法，通过将稀有和未知词编码为子词单元序列，使NMT模型能够进行词汇翻译。这是基于这样的直觉，即可以通过比单词小的单位来翻译各种单词类别，例如名称（通过字符复制或音译），复合词（通过成分翻译）以及同源词和借词（通过语音和词法转换）。
- 实验结果：我们讨论了包括简单字符ngram模型和基于字节对编码压缩算法的分段在内的不同分词技术的适用性，并根据经验表明，针对WMT 15翻译任务，英语→德语中子词模型在退避词典基线上有所改进和英语→俄语，最多分别为1.1和1.3 BLEU。

## 动机

- NMT模型的词汇一般是30000-50000，但是翻译却是open-vocabulary的问题。很多语言富有信息创造力，比如凝聚组合等等，翻译系统就需要一种低于word-level的机制。

- Word-level NMT的缺点
  - 对于word-level的NMT模型，翻译out-of-vocabulary的单词会回退到dictionary里面去查找。有下面几个缺点
    - 种技术在实际上使这种假设并不成立。比如源单词和目标单词并不是一对一的，你怎么找呢
    - 不能够翻译或者产生未见单词
    - 把unknown单词直接copy到目标句子中，对于人名有时候可以。但是有时候却需要改变形态或者直译。

## 本文工作

建立open-vocabulary的翻译模型，不用针对稀有词汇去查字典。事实证明，subword模型效果比传统大词汇表方法更好、更精确。Subword神经网络模型可以从subword表达中学习到组合和直译等能力，也可以有效的产生不在训练数据集中的词汇。

## 论文贡献

- open-vocabulary的问题可以通过对稀有词汇使用subword units单元来编码解决
- 采用Byte pair encoding (BPE) 算法来进行分割。BPE通过一个固定大小的词汇表来表示开放词汇，这个词汇表里面的是变长的字符串序列。这是一种对于神经网络模型非常合适的词分割策略。

## 论文思路

- 初始化符号词表。用所有的字符加入到符号词表中。对所有单词的末尾加入特殊标记，如-。翻译后恢复原始的标记。
- 迭代对所有符号进行计数，找出次数最多的(A, B)，用AB代替。
- 每次合并，会产生一个新的符号，代表着n-gram字符
- 常见的n-grams字符(或者whole words)，最终会被合并到一个符号
- 最终符号词表大小=初始大小+合并操作次数。操作次数是算法唯一的超参数。

不用考虑不在训练集里面的pair，为每个word根据出现频率设置权重。

和传统的压缩算法(哈夫曼编码)相比，我们的以subword 单元堆积的符号序列依然是可以解释的，网络也可以翻译和产生新的词汇（训练集没有见过的）。

## 参考资料

1. [Neural Machine Translation of Rare Words with Subword Units](https://plmsmile.github.io/2017/10/19/subword-units/)

