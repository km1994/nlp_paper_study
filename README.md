# NLP 小小白研读论文记

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 目录


- [NLP 小小白研读论文记](#nlp-小小白研读论文记)
  - [目录](#目录)
  - [介绍](#介绍)
    - [论文工具篇](#论文工具篇)
    - [会议收集篇](#会议收集篇)
    - [NLP 学习篇](#nlp-学习篇)
      - [理论学习篇](#理论学习篇)
        - [经典论文研读篇](#经典论文研读篇)
        - [transformer 学习篇](#transformer-学习篇)
        - [预训练模型篇](#预训练模型篇)
        - [细粒度情感分析论文研读](#细粒度情感分析论文研读)
        - [主动学习论文研读](#主动学习论文研读)
        - [对抗训练论文研读](#对抗训练论文研读)
        - [实体关系联合抽取论文研读：](#实体关系联合抽取论文研读)
        - [GCN 在 NLP 上的应用 论文研读：](#gcn-在-nlp-上的应用-论文研读)
        - [命名实体识别论文研读：](#命名实体识别论文研读)
        - [关系抽取论文研读：](#关系抽取论文研读)
        - [文本预处理](#文本预处理)
        - [问答系统论文学习](#问答系统论文学习)
        - [文本摘要论文学习](#文本摘要论文学习)
        - [文本匹配论文学习](#文本匹配论文学习)
        - [机器翻译论文学习](#机器翻译论文学习)
        - [对话系统论文学习](#对话系统论文学习)
      - [视频学习篇](#视频学习篇)
      - [实战篇](#实战篇)
    - [Elastrsearch 学习篇](#elastrsearch-学习篇)
    - [推荐系统 学习篇](#推荐系统-学习篇)
    - [竞赛篇](#竞赛篇)
    - [GCN_study学习篇](#gcn_study学习篇)
    - [ML 小白入门篇](#ml-小白入门篇)
    - [Java 实战篇](#java-实战篇)
    - [百度百科 ES 全文检索平台构建 实战篇](#百度百科-es-全文检索平台构建-实战篇)
    - [面试篇](#面试篇)
      - [Leetcode 篇](#leetcode-篇)
      - [DeepLearning-500-questions](#deeplearning-500-questions)
    - [大数据 实战篇](#大数据-实战篇)
      - [Spark 实战篇](#spark-实战篇)
    - [资源篇](#资源篇)
    - [CV 入门 实战篇](#cv-入门-实战篇)



## 介绍

### [论文工具篇](论文学习idea/)

- 问题
  - 作为一名 scholar，你是否和我一样，在刚入门 NLP 时，对于陌生领域有种无从下手，心存畏惧？
  - 作为一名 scholar，你是否还在发愁如何找好的论文？
  - 作为一名 scholar，你是否还在为 自己 的 英文阅读 能力跟不上 很烦恼？
  - 作为一名 scholar，你是否还在为 看到 一篇好paper，但是复现不出 code 而心累？
  - 作为一名 scholar，你是否还在为 有Good idea，Outstanding Experimental results，Beautiful Chinese manuscript，结果 Bad English manuscript, Poor Journal 而奔溃？
  - 作为一名 scholar，你是否在为搞科研没人交流而自闭？
- 当你看到这一篇文档，你将不在为这些问题而烦恼，因为我们为你准备了一整套免费的从 论文查找->论文翻译->论文理解->相关代码搜索->写英文稿->科研学术交流 的路径。
  - [论文不会找怎么办？](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#论文不会找怎么办)
    - [顶会资讯](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#顶会资讯)
    - [论文搜索和分析工具](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#论文搜索和分析工具)
  - [外文读不懂怎么办？](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#外文读不懂怎么办)
    - [论文翻译神器 ———— 通天塔](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#论文翻译神器--通天塔)
    - [论文翻译小助手 ———— 彩云小译](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#论文翻译小助手--彩云小译)
  - [外文没 code 怎么办？](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#外文没-code-怎么办)
    - [papers with code](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#papers-with-code) 
    - [OpenGitHub 新项目快报](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#opengithub-新项目快报) 
  - [外文写起来麻烦怎么办](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#外文写起来麻烦怎么办) 
    - [Overleaf](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#overleaf) 
    - [Authorea](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#authorea) 
    - [Code ocean](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#code-ocean) 
  - [搞科研没人交流怎么办？](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#搞科研没人交流怎么办) 
    - [Shortscience](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#shortscience) 
    - [OpenReview](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#openreview) 
    - [Scirate](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#scirate) 

### 会议收集篇
- [ACL2020](ACL/ACL2020.md)

### NLP 学习篇

#### 理论学习篇

##### 经典论文研读篇

- 那些你所不知道的事
  - [【关于Transformer】 那些的你不知道的事](transformer_study/Transformer/)
  - [【关于Bert】 那些的你不知道的事](bert_study/T1_bert/)

##### transformer 学习篇

- [transformer_study](transformer_study/)  transformer 论文学习
  - [【关于Transformer】 那些的你不知道的事](transformer_study/Transformer/)
  - [Transformer-XL](transformer_study/T3_Transformer_XL/)
  - [Single Headed Attention RNN: Stop Thinking With Your Head 单头注意力 RNN: 停止用你的头脑思考](transformer_study/SHA_RNN_study/)
  - [ Universal Transformers](transformer_study/T4_Universal_Transformers/)
  - [Style_Transformer](Style_Transformer/LCNQA/)
  - [ACL2020_Linformer](transformer_study/ACL2020_Linformer)

##### 预训练模型篇

- [bert_study](bert_study/)：Bert论文研读
  - [【关于Bert】 那些的你不知道的事](bert_study/T1_bert/)
  - [XLNet Generalized Autoregressive Pretraining for Language Understanding](bert_study/T2_XLNet/)
  - [RoBERTa: A Robustly Optimized BERT Pretraining Approach](bert_study/T4_RoBERTa/)
  - [ A lite BERT for self-supervised learning of language representations](bert_study/T5_ALBERT/)
  - [FastBERT](bert_study/FastBERT/)
  - [distilbert](bert_study/distilbert/)
  - [Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT](bert_study/ACL2020_UnsupervisedBert/)
  - [GRAPH-BERT: Only Attention is Needed for Learning Graph Representations]([bert_study/ACL2020_UnsupervisedBert/](https://github.com/km1994/nlp_paper_study/bert_study/T2020_GRAPH_BERT))

##### [细粒度情感分析论文研读](ABSC_study/)

- [LCF](ABSC_study/LCF/): A Local Context Focus Mechanism for Aspect-Based Sentiment Classiﬁcation
  
##### [主动学习论文研读](active_learn_study/)

- [Proactive Learning for Named Entity Recognition（命名实体识别的主动学习）](active_learn_study/ProactiveLearningforNamedEntityRecognition/)

##### [对抗训练论文研读](adversarial_training_study/)

- [FreeLB: Enhanced Adversarial Training for Language Understanding 加强语言理解的对抗性训练](adversarial_training_study/FREELB/)

##### [实体关系联合抽取论文研读](ERE_study/)：
- [Incremental Joint Extraction of Entity Mentions and Relations](ERE_study/T2014_joint_extraction/)
- [Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy](ERE_study/JointER/)
- [GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction](ERE_study/ACL2019_GraphRel/)
- [A Novel Hierarchical Binary Tagging Framework for Relational Triple Extraction](ERE_study/T20ACL_HBT_su/)

##### [GCN 在 NLP 上的应用 论文研读](GCN2NLP/)：
- [GCN 在 NLP 上的应用 论文研读](GCN2NLP/readme.md)

##### [命名实体识别论文研读](NER_study/)：
- [LatticeLSTM](NER_study/ACL2018_LatticeLSTM/)
- [named entity recognition using positive-unlabeled learning](NER_study/JointER/)
- [GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction](NER_study/ACL2019_NERusingPositive-unlabeledLearning/)
- [TENER: Adapting Transformer Encoder for Name Entity Recognition](NER_study/ACL2019_TENER/)
- [CrossWeigh从不完善的注释中训练命名实体标注器](NER_study/EMNLP2019/CrossWeigh从不完善的注释中训练命名实体标注器/)
- [利用词汇知识通过协同图网络进行中文命名实体识别](NER_study/EMNLP2019/利用词汇知识通过协同图网络进行中文命名实体识别/)
- [一点注释对引导低资源命名实体识别器有很多好处](NER_study/EMNLP2019/一点注释对引导低资源命名实体识别器有很多好处/)
- [CGN: Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network（EMNLP2019）](NER_study/EMNLP2019_CGN/)
- [Fine-Grained Entity Typing in Hyperbolic Space（在双曲空间中打字的细粒度实体）](NER_study/Fine-GrainedEntityTypinginHyperbolicSpace/)
- [LR-CNN:CNN-Based Chinese NER with Lexicon Rethinking(IJCAI2019)](NER_study/IJCAI2019_LR_CNN/)

##### [关系抽取论文研读](NRE_paper_study/)：
- [End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures【2016】](NER_study/T2016_LSTM_Tree/)
- [ERNIE](NRE_paper_study/ERNIE/)
- [GraphRel](NRE_paper_study/GraphRel/)
- [R_BERT](NRE_paper_study/R_BERT)
- [Task 1：全监督学习](NRE_paper_study/T1_FullySupervisedLearning/)
  - [Relation Classification via Convolutional Deep Neural Network](NRE_paper_study/T1_FullySupervisedLearning/T1_Relation_Classification_via_CDNN/)
  - [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](NRE_paper_study/T1_FullySupervisedLearning/T2_Attention-Based_BiLSTM_for_RC/)
  - [Relation Classification via Attention Model](NRE_paper_study/T1_FullySupervisedLearning/T3_RC_via_attention_model_new/)
- [Task 2：远程监督学习](NRE_paper_study/T2_DistantSupervisedLearning/)
  - [Relation Classification via Convolutional Deep Neural Network](NRE_paper_study/T2_DistantSupervisedLearning/T1_Piecewise_Convolutional_Neural_Networks/)
  - [NRE_with_Selective_Attention_over_Instances](NRE_paper_study/T2_DistantSupervisedLearning/T2_NRE_with_Selective_Attention_over_Instances/)
  
##### [文本预处理](pre_study/)
- [过采样]([pre_study/](https://github.com/km1994/nlp_paper_study/tree/master/pre_study/samplingStudy))

##### [问答系统论文学习](QA_study/) 
- [Lattice CNNs for Matching Based Chinese Question Answering](QA_study/LCNQA/)
- [LSTM-based Deep Learning Models for Non-factoid Answer Selection](QA_study/T1_LSTM-based_for_Non-factoid_Answer_Selection/)
- [Denoising Distantly Supervised Open-Domain Question Answering](QA_study/T4_DenoisingDistantlySupervisedODQA/)
- [FAQ retrieval using query-question similarity and BERT-based query-answer relevance](QA_study/ACM2019_faq_bert-based_query-answer_relevance/)

##### [文本摘要论文学习](summarization_study/) 
- [Fine-tune BERT for Extractive Summarization](summarization_study/EMNLP2019_bertsum/)
- [Pointer-Generator Networks 指针网络读书笔记](summarization_study/ACL2017_Pointer_Generator_Networks/)

##### [文本匹配论文学习](text_match_study/) 
- [Multi-Perspective Sentence Similarity Modeling with Convolution Neural Networks](text_match_study/Multi-PerspectiveSentenceSimilarityModelingwithCNN/)
- [Simple and Effective Text Matching with Richer Alignment Features](text_match_study/Multi-RE2_study/)
- [Deep Structured Semantic Model](text_match_study/cikm2013_DSSM/)
- [ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](text_match_study/TACL2016_ABCNN/)
- [Enhanced LSTM for Natural Language Inference](text_match_study/TACL2017_ESIM/)
- [Bilateral Multi-perspective Matching](text_match_study/IJCAI2017_BiMPM/)
- [Densely Interactive Inference Network（DIIN）](text_match_study/T2017_DIIN/)

##### [机器翻译论文学习](MachineTranslation/)

- [Neural Machine Translation of Rare Words with Subword Units 论文学习](MachineTranslation/NeuralMachineTranslationOfRareWordsWithSubwordUnits/)

##### [对话系统论文学习](DialogueSystem_study/)

1. [【关于 Domain/Intent Classification 】那些你不知道的事](IntentClassification/)
2. [【关于 槽位填充 (Slot Filling)】那些你不知道的事](SlotFilling/)
3. [【关于 上下文LU】那些你不知道的事](contextLU/)
4. [【关于 自然语言生成NLG 】那些你不知道的事](NLG/)
5. [【关于 DSTC 】那些你不知道的事](DSTC/)
6. [【关于 E2E 】那些你不知道的事](E2E/)
   1. [【关于 TC_Bot(End-to-End Task-Completion Neural Dialogue Systems) 】那些你不知道的事](E2E/TC_Bot/)


#### 视频学习篇

- [CS224n 视频学习篇](https://github.com/km1994/Datawhale_NLP_CS224n)
  -  [Lecture 1: Introduction and Word Vectors](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture1)
  -  [Lecture 2: Word Vectors and Word Senses](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture2)
  -  [Lecture 3: Word Window Classification, Neural Networks, and Matrix Calculus](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture3)
  -  [Lecture 4: Backpropagation](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture4)
  -  [Lecture 5: Dependency Parsing](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture5)
  -  [Lecture 6: Language Models and RNNs](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture6)
  -  [Lecture 7: Vanishing Gradients, Fancy RNNs](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture7)
  -  [Lecture 8: Translation, Seq2Seq, Attention](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture8)
  -  [Lecture 9: Practical Tips for Projects](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture9)
  -  [Lecture 10: Question Answering](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture10)
  -  [Lecture 11: Convolutional Networks for NLP](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture11)
  -  [Lecture 12: Subword Models](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture12)
  -  [Lecture 13: Contextual Word Embeddings](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture13)
  -  [Lecture 14: Transformers and Self-Attention](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture14)
  -  [Lecture 15: Natural Language Generation](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture15)
  -  [Lecture 16: Coreference Resolution](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture16)
  -  [Lecture 17: Multitask Learning](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture17)
  -  [Lecture 18: Constituency Parsing, TreeRNNs](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture18)
  -  [Lecture 19: Bias in AI](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture19)
  -  [Lecture 20: Future of NLP + Deep Learning](https://github.com/km1994/Datawhale_NLP_CS224n/tree/master/Lecture/Lecture20)

#### 实战篇

- [爬虫 实战篇](https://github.com/km1994/Conversation_Wp/tree/master/t11_scrapyWp)
  - [Scrapy 爬虫 实战篇](https://github.com/km1994/Conversation_Wp/tree/master/t11_scrapyWp):主要介绍使用 scrapy 构建网络爬虫，并爬去百度搜索引擎数据

- [特征提取 实战篇](https://github.com/km1994/text_feature_extraction)
  - [关键词提取、关键短语提取、文本摘要提取 实战篇](https://github.com/km1994/text_feature_extraction)
  - [TF-idf 特征提取 实战篇](https://github.com/km1994/text_feature_extraction)
  - [pynlp 关键词提取 实战篇](https://github.com/km1994/Conversation_Wp/tree/master/t10_pynlpirWp)

- [词向量预训练 实战篇](https://github.com/km1994/TextClassifier/tree/master/word2vec_train)
  - [word2vec 词向量预训练 实战篇](https://github.com/km1994/TextClassifier/tree/master/word2vec_train)
  - [fasttext 词向量预训练 实战篇](https://github.com/km1994/TextClassifier/tree/master/word2vec_train)

- [中文情感分析 实战篇](https://github.com/km1994/sentiment_analysis)
  - [word2vec](https://github.com/km1994/sentiment_analysis/tree/master/word2vec)
  - [textCNN](https://github.com/km1994/sentiment_analysis/tree/master/textCNN)
  - [charCNN](https://github.com/km1994/sentiment_analysis/tree/master/charCNN)
  - [RCNN](https://github.com/km1994/sentiment_analysis/tree/master/RCNN)
  - [Bi-LSTM](https://github.com/km1994/sentiment_analysis/tree/master/Bi-LSTM)
  - [Bi-LSTM+Attention](https://github.com/km1994/sentiment_analysis/tree/master/Bi-LSTM%2BAttention)
  - [adversarialLSTM](https://github.com/km1994/sentiment_analysis/tree/master/adversarialLSTM)
  - [Transformer](https://github.com/km1994/sentiment_analysis/tree/master/Transformer)
  - [ELMo](https://github.com/km1994/sentiment_analysis/tree/master/ELMo)
  - [BERT](https://github.com/km1994/sentiment_analysis/tree/master/BERT)

- [中文文本分类 实战篇](https://github.com/km1994/TextClassifier)
  - [Tensorflow 篇](https://github.com/km1994/TextClassifier)
    - [FastText](https://github.com/km1994/TextClassifier/tree/master/fastTextStudy.ipynb)
    - [TextCNN](https://github.com/km1994/TextClassifier/tree/master/dl_model)
    - [TextRNN](https://github.com/km1994/TextClassifier/tree/master/dl_model)
    - [TextRCNN](https://github.com/km1994/TextClassifier/tree/master/dl_model)
    - [BiLSTMAttention](https://github.com/km1994/TextClassifier/tree/master/dl_model)
    - [AdversarialLSTM](https://github.com/km1994/TextClassifier/tree/master/dl_model)
    - [Transformer](https://github.com/km1994/TextClassifier/tree/master/dl_model)
  - [pytorch 篇](https://github.com/km1994/Chinese-Text-Classification-Pytorch)
    - [FastText](https://github.com/km1994/Chinese-Text-Classification-Pytorch)
    - [TextCNN](https://github.com/km1994/Chinese-Text-Classification-Pytorch)
    - [TextRNN](https://github.com/km1994/Chinese-Text-Classification-Pytorch)
    - [TextRCNN](https://github.com/km1994/Chinese-Text-Classification-Pytorch)
    - [BiLSTMAttention](https://github.com/km1994/Chinese-Text-Classification-Pytorch)
    - [DPCNN](https://github.com/km1994/Chinese-Text-Classification-Pytorch)
    - [AdversarialLSTM](https://github.com/km1994/Chinese-Text-Classification-Pytorch)
    - [Transformer](https://github.com/km1994/Chinese-Text-Classification-Pytorch)

- [命名实体识别 “史诗级” 入门教程](https://github.com/km1994/NERer)
  - [HMM 做命名实体识别](https://github.com/km1994/named_entity_recognition/models/hmm.py)
  - [CRF 做命名实体识别](https://github.com/km1994/named_entity_recognition/models/crf.py)
  - [BiLSTM-CRF 做命名实体识别](https://github.com/km1994/NERer/tree/master/LSTM_IDCNN)
  - [IDCNN-CRF 做命名实体识别](https://github.com/km1994/NERer/tree/master/LSTM_IDCNN)
  - [BERT-CRF 做命名实体识别](https://github.com/km1994/NERer/tree/master/bert_crf)
  - [ALBERT-CRF 做命名实体识别](https://github.com/km1994/NERer/tree/master/albert_crf)

- [知识图谱 实战篇]()
  - [KBQA-BERT](https://github.com/km1994/KBQA-BERT)

- [问答系统 实战篇](https://github.com/km1994/Conversation_Wp/tree/master/baidu_qa_zh_process)
  - [基于 百度问答 的问答系统](https://github.com/km1994/Conversation_Wp/tree/master/baidu_qa_zh_process/Baidu_WebQA_model.ipynb)

- [文本匹配 实战篇](https://github.com/km1994/TextMatching)
  - [TextMatching](https://github.com/km1994/TextMatching)
  - [TextMatch](https://github.com/km1994/TextMatch)
  - [Text_Matching（文本匹配算法）](https://github.com/km1994/Text_Matching)

- [预训练模型 实战篇]()
  - [bert](https://github.com/km1994/bert)
  - [Chinese-PreTrained-XLNet](https://github.com/km1994/Chinese-PreTrained-XLNet)

- [模型蒸馏 实战篇]()
  - [基于BERT的蒸馏实验](https://github.com/km1994/bert_distill)

### Elastrsearch 学习篇

- [Elastrsearch 学习](es_study/)
  - [ElasticSearch架构解析与最佳实践.md](es_study/ElasticSearch架构解析与最佳实践.md)

### 推荐系统 学习篇

- [推荐系统 论文学习](RecommendedSystem_study)
  - [DeepFM 论文学习](RecommendedSystem_study/DeepFM_study)
  - [DeepWalk 论文学习](RecommendedSystem_study/DeepWalk)
  - [ESMM 论文学习](RecommendedSystem_study/ESMM_study)
  - [【关于 FiBiNET】那些你不知道的事](RecommendedSystem_study/FiBiNet_study)
  - [【关于 DeepCF】那些你不知道的事](RecommendedSystem_study/DeepCF_study)
  
### 竞赛篇

- [竞赛篇](game_study)

### [GCN_study学习篇](https://github.com/km1994/GCN_study)

- GCN 介绍篇
  - [Graph 介绍](https://github.com/km1994/GCN_study/blob/master/graph_introduction/graph_introduction.md)
  - [Weisfeiler-Leman 算法介绍](https://github.com/km1994/GCN_study/blob/master/WL/WL_conclusion.md)

- GCN 三剑客
  - [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://github.com/km1994/GCN_study/blob/master/CNNonGraph_Defferrard_2016/CNNonGraph_Defferrard_2016_总结.md)
  - [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://github.com/km1994/GCN_study/blob/master/GCN/SEMI-SUPERVISED%20CLASSIFICATION%20WITH%20GRAPH%20CONVOLUTIONAL%20NETWORKS.pdf)
  - [Attention Models in Graphs: A Survey](https://github.com/km1994/GCN_study/blob/master/Attention_models/Attention_models.md)

- 经典篇
  - [Can GNN go “online”?an analysis of pretraining and inference](https://github.com/km1994/GCN_study/tree/master/CanGNNGoOnline)
  - [Graph Convolutional Networks for Text Classification](https://github.com/km1994/GCN_study/tree/master/GCNforTextClassification)
  - [HOW POWERFUL ARE GRAPH NEURAL NETWORKS](https://github.com/km1994/GCN_study/blob/master/HowPowerGCN/HOW%20POWERFUL%20ARE%20GRAPH%20NEURAL%20NETWORKS.pdf)
  - [Graph Convolutional Matrix Completion](https://github.com/km1994/GCN_study/blob/master/GCMC/Graph%20Convolutional%20Matrix%20Completion.pdf)
  - [Representation Learning For Attributed Multiplex Heterogeneous Network](https://github.com/km1994/GCN_study/blob/master/RepresentationLearningForAttributedMultiplexHeterogeneousNetwork/RepresentationLearningForAttributedMultiplexHeterogeneousNetwork.md)

- 预训练篇
  - [GNN 教程：GCN 的无监督预训练](https://github.com/km1994/GCN_study/tree/master/pretrainingGCN_1)
  - [Pre-training Graph Neural Networks](https://github.com/km1994/GCN_study/blob/master/pretrainingGCN_2/pretrainingGCN.md)

- 实战篇
  - [DGL](https://github.com/km1994/dgl)
  - [DGL 入门](https://github.com/km1994/GCN_study/blob/master/DGL_study/DGL_introduction.md)
  - [DGL 入门 —— GCN 实现](https://github.com/km1994/GCN_study/blob/master/DGL_study/DGL_GCN_introduction.md)

### [ML 小白入门篇](https://github.com/km1994/ML_beginer)

- [概率图模型串烧 （HMM->MEMM->CRF）](https://github.com/km1994/ML_beginer/tree/master/CRF_study)

- [KNN 算法 学习篇](https://github.com/km1994/MLStudy#%E4%B8%80knnstudy)
  - [理论篇](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000006&idx=7&sn=f040e0a95376880349378ce7afd634af&chksm=1bbfe43c2cc86d2a5942e0938c871375cffcd72c9795a3e8b39fdb0c8285348631e7bf8a27b4)
  - [实战篇](https://github.com/km1994/MLStudy/blob/master/KNNStudy/KNNClass.py)

- [朴素贝叶斯算法 学习篇](https://github.com/km1994/MLStudy#%E4%BA%8Cnbstudy)
  - [NB 算法理论学习](https://github.com/km1994/MLStudy/blob/master)
  - [NB 算法实现](https://github.com/km1994/MLStudy/blob/master/NBStudy/NBStudy.py)

- [Apriori 算法 学习篇](https://github.com/km1994/MLStudy#%E4%B8%89aprioristudy)
  - [Apriori 算法理论学习](https://github.com/km1994/MLStudy/blob/master)
  - [Apriori 算法实现](https://github.com/km1994/MLStudy/blob/master/AprioriStudy/AprioriMyTest.py)

- [Softmax 算法学习篇](https://github.com/km1994/MLStudy#%E5%8D%81softmax-numpy-%E5%AE%9E%E7%8E%B0)
  - [Softmax 理论学习](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001925&idx=5&sn=20c5ead4f4b5f8f88c30043fb3703557&chksm=1bbfedbf2cc864a96b5fc4575e09294478f6f4cff65d654b8d775fed78766f80faf333d8ca08&scene=20&xtrack=1#rd)
  - [Softmax 算法实现](https://github.com/km1994/MLStudy/blob/master/softmaxStudy/softmaxStudy.py)

- [Gradient Descent 算法学习篇](https://github.com/km1994/MLStudy#%E5%9B%9Bgradientdescentstudy)
  - [GradientDescent 算法理论学习](https://github.com/km1994/MLStudy/blob/master)
  - [GradientDescent 算法实现](https://github.com/km1994/MLStudy/blob/master/GradientDescentStudy/GradientDescentTest.py)

- [随机森林算法 学习篇](https://github.com/km1994/MLStudy#%E4%BA%94randomforest)
  - [RandomForest 算法理论学习](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001388&idx=1&sn=21bae727bf3510fad98b3ec4a89d124e&chksm=1bbfe3562cc86a40769ea726f96e3a45185697f9582a2e3fbbbeec3af90dd722ebe09b635ddc&scene=20&xtrack=1#rd)
  - [RandomForest 算法实现](https://github.com/km1994/MLStudy/blob/master/RandomForest/xiechengRF/RandomForestClass.py)
  - [基于PCA 的 RandomForest 算法实现](https://github.com/km1994/MLStudy/blob/master/RandomForest/xiechengRF/xiechengPCARF.py)

- [EM 算法学习篇](https://github.com/km1994/ML_beginer/tree/master/EM_study)

- [SVM 算法学习篇](https://github.com/km1994/MLStudy#%E5%85%ADsvn)
  - [SVN 算法理论学习](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100001388&idx=1&sn=21bae727bf3510fad98b3ec4a89d124e&chksm=1bbfe3562cc86a40769ea726f96e3a45185697f9582a2e3fbbbeec3af90dd722ebe09b635ddc&scene=20&xtrack=1#rd)
  - [SVM 算法学习篇](https://github.com/km1994/ML_beginer/tree/master/SVM_study)
  - [SVN 算法实现](https://github.com/km1994/MLStudy/blob/master/SVN/PCAandSVN.py)

- [BPNN 算法 学习篇](https://github.com/km1994/MLStudy#%E4%B8%83bpnn)
  - [BPNN 算法理论学习](https://github.com/km1994/MLStudy/blob/master)
  - [BPNN 算法实现](https://github.com/km1994/MLStudy/blob/master/ANN/BP/test.py)

- [PCA 算法 学习篇](https://github.com/km1994/MLStudy#%E5%85%ABpca)
  - [PCA 算法理论学习](https://github.com/km1994/MLStudy/blob/master)
  - [PCA 算法实现](https://github.com/km1994/MLStudy/blob/master/DimensionalityReduction/PCA/PCAClass.py)

- [CNN 算法 学习篇](https://github.com/km1994/MLStudy#%E4%B9%9Dcnn-numpy-%E5%AE%9E%E7%8E%B0)
  - [卷积运算的定义、动机](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000973&idx=2&sn=4eced9e7204274a4e3798cb73a140c72&chksm=1bbfe1f72cc868e1f63262fad39b4f735a6d424c064f6bee755e438b94487bf75b5d41cc02c0&scene=20&xtrack=1&key=fe048f5ad4fa1bcff1ed72e320faab18cb01c02c1a16279c60775734b428e42206e9f5a8f3c402ae96c01259df639ca04206da43e2ab1b42bfaf44bb4068c793df27faa893ea0301a375086e4adfd3b7&ascene=1&uin=MjQ3NDIwNTMxNw%3D%3D&devicetype=Windows+10&version=62060426&lang=zh_CN&pass_ticket=906xy%2Fk9TQJp5jnyekYc89wLa17ODmZRkas9HXdX%2BtYcy0q32NIxLHOhFx519Yxa)
  - [反卷积Deconvolution](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000973&idx=3&sn=8a787cc0e165fa071ca7a602f16fae17&chksm=1bbfe1f72cc868e1249a3ebe90021e2a6e12c3d8021fcc1877a5390eed36f5a8a6698eb65216&scene=20&xtrack=1#rd)
  - [池化运算的定义](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100000973&idx=4&sn=cebf71790dd7e0e497fa36fa199c368d&chksm=1bbfe1f72cc868e1017d26a996f1eb7602fad1efced62713def3012b8df1b85b6ba46c0ebae8&scene=20&xtrack=1#rd)
  - [CNN 算法 numpy 实现](https://github.com/km1994/MLStudy/blob/master/CNN-Numpy_suss/view.py)

### [Java 实战篇](https://github.com/km1994/3y)

### [百度百科 ES 全文检索平台构建 实战篇](https://github.com/km1994/baidu_es)

- 项目目标
  - 实现一个 基于 百度百科 的 ES 全文检索平台
- 项目流程
  - step 1: 编写 网络爬虫 爬取 百度百科 数据；
  - step 2: 爬取数据之后，需要进行预处理操作，清洗掉 文本中噪声数据；
  - step 3: 将输入 导入 ES ；
  - step 4: 利用 python 编写 后台，并 对 ES 进行查询，返回接口数据；
  - step 5: ES 数据前端展示；
  - step 6: 百度百科 知识图谱构建
  - step 7：百度百科 知识图谱检索与展示
- 数据介绍：
  - 本项目通过编写爬虫爬取 百度百科 数据，总共爬取 名称、链接、简介、中文名、外文名、国籍、出生地、出生日期、职业、类型、中文名称、代表作品、民族、主要成就、别名、毕业院校、导演、制片地区、主演、编剧、上映时间 等400多个 指标，共爬取数据 98000 条。
- 数据预处理模块

爬取的数据根据名称可以分为 人物、地点、书籍、作品、综艺节目等。

|    类别    | 指标量  | 数量   | 筛选方式  |
| :--------: | :----: | :----: |  :----:  |
|    人物    |   109  |  27319  | 国籍、职业、出生日期、出生地有一个不为空 |
|    地点    |   124  |   9361  | 地理位置、所属地区有一个不为空 |
|    书籍    |   45   |   3336  | 作者 不为空 |
|    作品    |   45   |   8850  | 主演为空，中文名称不为空 |
|  综艺节目  |   108  |   5600  | 主演、导演都不为空 |

### [面试篇]()

#### [Leetcode 篇](https://github.com/km1994/leetcode/blob/master/README.md)

- [简单题](https://github.com/km1994/leetcode/blob/master/topic1_easy/)
- [数组](https://github.com/km1994/leetcode/blob/master/topic2_arr/)
- [链表](https://github.com/km1994/leetcode/blob/master/topic3_List/)
- [动态规划](https://github.com/km1994/leetcode/blob/master/topic4_dynamic_planning_study/)
- [字符串](https://github.com/km1994/leetcode/blob/master/topic5_string/)
- [栈](https://github.com/km1994/leetcode/blob/master/topic6_stack/)
- [排序](https://github.com/km1994/leetcode/blob/master/topic7_sorted/)
- [二分查找](https://github.com/km1994/leetcode/blob/master/topic8_binary_search/)
- [哈希表](https://github.com/km1994/leetcode/blob/master/topic9_hash_table/)
- [队列](https://github.com/km1994/leetcode/blob/master/topic10_queue/)
- [堆](https://github.com/km1994/leetcode/blob/master/topic11_heap/)
- [回溯法](https://github.com/km1994/leetcode/blob/master/topic12_backtrack/)
- [树](https://github.com/km1994/leetcode/blob/master/topic13_tree/)
- [归并排序](https://github.com/km1994/leetcode/blob/master/topic15_merge/)
- [快慢指针](https://github.com/km1994/leetcode/blob/master/topic16_speed_pointer/)
- [贪心算法](https://github.com/km1994/leetcode/blob/master/topic17_greedy/)
- [递归](https://github.com/km1994/leetcode/blob/master/topic18_recursive/)
- [分治](https://github.com/km1994/leetcode/blob/master/topic19_divide_and_conquer/)
- [分支限界法](https://github.com/km1994/leetcode/blob/master/topic20_branch_and_bound_method/)
- [位运算](https://github.com/km1994/leetcode/blob/master/topic21_Bit_operation/)
- [滑动窗口](https://github.com/km1994/leetcode/blob/master/topic22_move_window/)
- [数学题](https://github.com/km1994/leetcode/blob/master/topic23_math/)
- [面试题](https://github.com/km1994/leetcode/blob/master/interview/)

#### [DeepLearning-500-questions](https://github.com/km1994/DeepLearning-500-questions)

### [大数据 实战篇]()
#### [Spark 实战篇](https://github.com/km1994/sparkStudy)
- 1、wordCount
  - 内容：运行原理，RDD设计，DAG，安装与使用
  - 第1章 Spark的设计与运行原理（大概了解）
    - 1.1 Spark简介
    - 1.2 Spark运行架构
    - 1.3 RDD的设计与运行原理
    - 1.4 Spark的部署模式
  - 第2章 Spark的安装与使用（主要内容）
    - 2.1 Spark的安装和使用 （如果想在window上安装，参考https://blog.csdn.net/SummerHmh/article/details/89518567，之后可以用pyspark或者jupyter上进行学习）（地址有问题，可以使用这个https://www-eu.apache.org/dist/spark/spark-2.4.3/）
    - 2.2 第一个Spark应用程序：WordCount

- 2、RDDStudy
  - 内容：RDD编程，熟悉算子，读写文件
  - 第3章 Spark编程基础
    - 3.1 Spark入门：RDD编程
    - 3.2 Spark入门：键值对RDD
    - 3.3 Spark入门：共享变量（提升-分布式必备）
    - 3.4 数据读写
      - 3.4.1 Spark入门：文件数据读写

- 3、sparkSQLStudy
  - 内容：DataFrame,SparkSQL
  - 第4章
    - 4.1 Spark SQL简介
    - 4.2 DataFrame与RDD的区别
    - 4.3 DataFrame的创建
    - 4.4 从RDD转换得到DataFrame

- 4、Parquet_JDBC_IO_Study

- 5、MLlibStudy
  - 内容：MLlib流设计，特征工程
  - 第6章 Spark MLlib
    - 6.1 Spark MLlib简介
    - 6.2 机器学习工作流
      - 6.2.1 机器学习工作流(ML Pipelines) 
      - 6.2.2 构建一个机器学习工作流
    - 6.3 特征抽取、转化和选择
      - 6.3.1 特征抽取：TF-IDF
      - 6.3.4 特征变换：标签和索引的转化
      - 6.3.5 特征选取：卡方选择器
  
### [资源篇](https://github.com/km1994/funNLP)
- [funNLP](https://github.com/km1994/funNLP)

### [CV 入门 实战篇](https://github.com/km1994/cv_entrance)


