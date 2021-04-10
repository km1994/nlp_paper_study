# 【关于 NLP比赛】 那些你不知道的事

## 目录

- [【关于 NLP比赛】 那些你不知道的事](#关于-nlp比赛-那些你不知道的事)
  - [目录](#目录)
  - [一、问答匹配任务](#一问答匹配任务)
    - [1. CCF2020问答匹配比赛](#1-ccf2020问答匹配比赛)
      - [1.1 比赛背景](#11-比赛背景)
      - [1.2 比赛方案收集](#12-比赛方案收集)
    - [2. 智能客服问题相似度算法设计——第三届魔镜杯大赛](#2-智能客服问题相似度算法设计第三届魔镜杯大赛)
    - [3. 2018CIKM AnalytiCup – 阿里小蜜机器人跨语言短文本匹配算法竞赛](#3-2018cikm-analyticup--阿里小蜜机器人跨语言短文本匹配算法竞赛)
    - [其他](#其他)
  - [二、对话](#二对话)
    - [1. 2020 CCF BDCI《千言：多技能对话》](#1-2020-ccf-bdci千言多技能对话)
      - [1.1 赛题简介](#11-赛题简介)
      - [1.2 比赛方案收集](#12-比赛方案收集-1)
    - [2. 2018JD Dialog Challenge 任务导向型对话系统挑战赛](#2-2018jd-dialog-challenge-任务导向型对话系统挑战赛)
  - [三、文本分类](#三文本分类)
    - [1. 2018 DC达观-文本智能处理挑战](#1-2018-dc达观-文本智能处理挑战)
    - [2. 路透社新闻数据集“深度”探索性分析(词向量/情感分析)](#2-路透社新闻数据集深度探索性分析词向量情感分析)
    - [3. 知乎看山杯](#3-知乎看山杯)
    - [4. 2018 CCL 客服领域用户意图分类评测](#4-2018-ccl-客服领域用户意图分类评测)
    - [5. 2018 kaggle quora insincere questions classification](#5-2018-kaggle-quora-insincere-questions-classification)
  - [四、 关键词提取](#四-关键词提取)
    - [1. “神策杯”2018高校算法大师赛(关键词提取)](#1-神策杯2018高校算法大师赛关键词提取)
  - [五、内容识别](#五内容识别)
    - [1. 第二届搜狐内容识别大赛](#1-第二届搜狐内容识别大赛)
  - [六、观点主题](#六观点主题)
    - [1. 汽车行业用户观点主题及情感识别](#1-汽车行业用户观点主题及情感识别)
  - [七、实体链指](#七实体链指)
    - [1. CCKS&百度 2019中文短文本的实体链指](#1-ccks百度-2019中文短文本的实体链指)
  - [参考资料](#参考资料)

## 一、问答匹配任务

### 1. CCF2020问答匹配比赛

#### 1.1 比赛背景

- 赛题名：房产行业聊天问答匹配
- 比赛链接：https://www.datafountain.cn/competitions/474

- 背景：贝壳找房是以技术驱动的品质居住服务平台，“有尊严的服务者、更美好的居住”，是贝壳的使命。在帮助客户实现更美好的居住过程中，客户会和服务者（房产经纪人）反复深入交流对居住的要求，这个交流发生在贝壳APP上的IM中。
IM交流是双方建立信任的必要环节，客户需要在这个场景下经常向服务者咨询许多问题，而服务者是否为客户提供了感受良好、解答专业的服务就很重要，贝壳平台对此非常关注。因此，需要准确找出服务者是否回答了客户的问题，并进一步判断回答得是否准确得体，随着贝壳平台规模扩大，需要AI参与这个过程。

- 任务：赛题任务：本次赛题的任务是：给定IM交流片段，片段包含一个客户问题以及随后的经纪人若干IM消息，从这些随后的经纪人消息中找出一个是对客户问题的回答。
任务要点：

1. 数据来自一个IM聊天交流过程；
2. 选取的客户问题之前的聊天内容不会提供；
3. 提供客户问题之后的经纪人发送的内容；
4. 如果在这些经纪人发送内容之间原本来穿插了其他客户消息，不会提供；
5. 这些经纪人发送内容中有0条或多条对客户问题的回答，把它找出来。

参赛者需要根据训练语料，构建出泛化能力强的模型，对不在训练语料中的测试数据做识别，从测试数据中为客户问题找出对应经纪人回答。希望参赛者能构建基于语义的识别模型，模型类型不限。

- 难度与挑战：

1. IM聊天的随意性和碎片化，各个地方的语言习惯不同。
2. 要求模型的泛化性好。在测试集上模型的度量指标。
3. 要求模型的复杂度小。最终提交模型需要符合生产环境使用要求。

- 出题单位：贝壳找房

#### 1.2 比赛方案收集

<table>
    <tr>
        <td>名次</td>
        <td>分数</td>
        <td>方案介绍</td>
        <td>github</td>
    </tr>
    <tr>
        <td>1</td>
        <td>A/0.81 B/0.830</td>
        <td>[方案介绍](https://xv44586.github.io/2021/01/20/ccf-qa-2/)</td>
        <td>[github](https://github.com/xv44586/ccf_2020_qa_match)</td>
    </tr>
</tabel>

### 2. 智能客服问题相似度算法设计——第三届魔镜杯大赛

- rank6 https://github.com/qrfaction/paipaidai
- rank12 https://www.jianshu.com/p/827dd447daf9 https://github.com/LittletreeZou/Question-Pairs-Matching
- Rank16：https://github.com/guoday/PaiPaiDai2018_rank16
- Rank29: https://github.com/wangjiaxin24/daguan_NLP

### 3. 2018CIKM AnalytiCup – 阿里小蜜机器人跨语言短文本匹配算法竞赛

- Rank2: https://github.com/zake7749/Closer
- Rank12：https://github.com/Leputa/CIKM-AnalytiCup-2018
- Rank18: https://github.com/VincentChen525/Tianchi/tree/master/CIKM%20AnalytiCup%202018

### 其他

- [Chinese sentence similarity](ChineseSentenceSimilarity/) 

## 二、对话

### 1. 2020 CCF BDCI《千言：多技能对话》

#### 1.1 赛题简介

- 赛题名称：千言：多技能对话
- 出题单位：百度
- 赛题背景

近年来，人机对话技术受到了学术界和产业界的广泛关注。学术上，人机对话是人机交互最自然的方式之一，其发展影响及推动着语音识别与合成、自然语言理解、对话管理以及自然语言生成等研究的进展；产业上，众多产业界巨头相继推出了人机对话技术相关产品，并将人机对话技术作为其公司的重点研发方向。以上极大地推动了人机对话技术在学术界和产业界的发展。

开放域对话技术旨在建立一个开放域的多轮对话系统，使得机器可以流畅自然地与人进行语言交互，既可以进行日常问候类的闲聊，又可以完成特定功能，以使得开放域对话技术具有实际应用价值，例如进行对话式推荐，或围绕一个主题进行深入的知识对话等。具体的说，开放域对话可以继续拆分为支持不同功能的对话形式，例如对话式推荐，知识对话技术等，如何解决并有效融合以上多个技能面临诸多挑战。

目前，学术界已经公开了多个面向开放域对话建模的开源数据集。但大多数研究工作仅关注模型在单一或少量数据集上的效果。尽管一些模型在单一数据集上取得了很好的效果，但缺乏在多个不同技能、不同领域数据上的评价，与真正很好的解决开放域对话这一技术挑战还有一定距离。为了解决这个问题，我们需要有一套评估全面，领域覆盖广的公开评测数据集。因此，本次竞赛主要基于百度千言数据集（https://luge.ai）及清华开放数据集（https://github.com/thu-coai/CDial-GPT），这些数据集收集了一系列公开的开放域对话数据，并对数据进行了统一的整理以及提供了统一的评测方式，期望从多个技能、多个领域的角度对模型效果进行综合评价。本次竞赛数据集旨在为研究人员和开发者提供学术和技术交流的平台，进一步提升开放域对话的研究水平，推动自然语言理解和人工智能领域技术的应用和发展。

- 赛题任务

本次评测的开放域对话数据集包含多个数据，涵盖了多个功能场景：包括日常闲聊对话，知识对话、推荐对话等。我们旨在衡量开放域对话模型在各个不同技能上的效果和模型通用性。

具体来说，本次比赛中我们主要从三个方面评测开放领域对话模型的能力：

1. 闲聊对话：在闲聊场景中，是否可以生成流畅的、与上下文相关的对话回复。
2. 知识对话：是否可以在对话过程中充分利用外部知识，并且在生成对话回复的过程中引入外部知识。
3. 推荐对话：是否可以在对话过程中基于用户兴趣以及用户的实时反馈，主动对用户做出推荐。

参赛队所构建的模型需要同时具备上述三项能力。

#### 1.2 比赛方案收集

<table>
    <tr>
        <td>名次</td>
        <td>分数</td>
        <td>方案介绍</td>
        <td>github</td>
    </tr>
    <tr>
        <td>1</td>
        <td>A/0.9266667</td>
        <td>[方案介绍](https://mp.weixin.qq.com/s/SdutOtTNJaKzlsozlhxxHA)</td>
        <td>[github](https://github.com/apple55bc/CCF-BDCI-qianyan)</td>
    </tr>
</tabel>

### 2. 2018JD Dialog Challenge 任务导向型对话系统挑战赛

- Rank2: https://github.com/Dikea/Dialog-System-with-Task-Retrieval-and-Seq2seq
- Rank3: https://github.com/zengbin93/jddc_solution_4th

## 三、文本分类

### 1. 2018 DC达观-文本智能处理挑战

- Rank1: https://github.com/ShawnyXiao/2018-DC-DataGrand-TextIntelProcess
- Rank2：https://github.com/CortexFoundation/-
- Rank4: https://github.com/hecongqing/2018-daguan-competition
- Rank8：https://github.com/Rowchen/Text-classifier
- Rank10: https://github.com/moneyDboat/data_grand 
- Rank11：https://github.com/TianyuZhuuu/DaGuan_TextClassification_Rank11
- Rank18: https://github.com/nlpjoe/daguan-classify-2018
- RankX: https://github.com/yanqiangmiffy/daguan

### 2. 路透社新闻数据集“深度”探索性分析(词向量/情感分析)

- https://www.kaggle.com/hoonkeng/deep-eda-word-embeddings-sentiment-analysis/notebook

### 3. 知乎看山杯

- Rank1：https://github.com/chenyuntc/PyTorchText
- Rank2：https://github.com/Magic-Bubble/Zhihu
- Rank6：https://github.com/yongyehuang/zhihu-text-classification 
- Rank9：https://github.com/coderSkyChen/zhihu_kanshan_cup_2017
- Rank21：https://github.com/zhaoyu87/zhihu

### 4. 2018 CCL 客服领域用户意图分类评测

- Rank1：https://github.com/nlpjoe/2018-CCL-UIIMCS

### 5. 2018 kaggle quora insincere questions classification 

- Rank1: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568 
- Rank13: https://mp.weixin.qq.com/s/DD-BOtPbGCXvxfFxL-qOgg 
- Rank153: https://github.com/jetou/kaggle-qiqc

## 四、 关键词提取

### 1. “神策杯”2018高校算法大师赛(关键词提取)

- Rank1: http://www.dcjingsai.com/common/bbs/topicDetails.html?tid=2382
- Rank2: https://github.com/bigzhao/Keyword_Extraction
- Rank5: https://github.com/Dikea/ShenceCup.extract_keywords

## 五、内容识别

### 1. 第二届搜狐内容识别大赛

- Rank1：https://github.com/zhanzecheng/SOHU_competition

## 六、观点主题

### 1. 汽车行业用户观点主题及情感识别 

- baseline 62+：https://github.com/312shan/Subject-and-Sentiment-Analysis

## 七、实体链指

### 1. CCKS&百度 2019中文短文本的实体链指

- [CCKS&百度 2019中文短文本的实体链指 第一名解决方案](https://github.com/panchunguang/ccks_baidu_entity_link)
- [CCKS 2019 中文短文本实体链指比赛技术创新奖解决方案](https://github.com/AlexYangLi/ccks2019_el)
- [多因子融合的实体识别与链指消歧](https://zhuanlan.zhihu.com/p/79389393)
- [论文导读 | OpenAI的实体消歧新发现](https://juejin.im/post/6844903566860091399)















## 参考资料

1. [Data competition Top Solution 数据竞赛Top解决方案开源整理](https://github.com/Smilexuhc/Data-Competition-TopSolution)
2. [nlp-competitions-list-review](https://github.com/zhpmatrix/nlp-competitions-list-review)


