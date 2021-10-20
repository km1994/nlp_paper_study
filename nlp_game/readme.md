# 【关于 NLP比赛】 那些你不知道的事

> 作者：杨夕
> 
> 论文：MLBiNet: A Cross-Sentence Collective Event Detection Network
> 
> 本文链接：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 
> 【注：手机阅读可能图片打不开！！！】

- [【关于 NLP比赛】 那些你不知道的事](#关于-nlp比赛-那些你不知道的事)
  - [一、问答匹配任务](#一问答匹配任务)
    - [5. 新冠疫情相似句对判定大赛 【比赛地址】](#5-新冠疫情相似句对判定大赛-比赛地址)
      - [5.1 赛题背景](#51-赛题背景)
      - [5.2 数据集介绍](#52-数据集介绍)
      - [5.3 比赛方案收集](#53-比赛方案收集)
    - [4. 2021搜狐校园文本匹配算法大赛 【比赛地址】](#4-2021搜狐校园文本匹配算法大赛-比赛地址)
      - [4.1 赛题背景](#41-赛题背景)
      - [4.2 比赛任务](#42-比赛任务)
      - [4.3 数据说明 【地址】](#43-数据说明-地址)
      - [4.4 比赛方案收集](#44-比赛方案收集)
    - [3. CCF2020问答匹配比赛](#3-ccf2020问答匹配比赛)
      - [3.1 比赛背景](#31-比赛背景)
      - [3.2 比赛方案收集](#32-比赛方案收集)
    - [2. 智能客服问题相似度算法设计——第三届魔镜杯大赛](#2-智能客服问题相似度算法设计第三届魔镜杯大赛)
    - [1. 2018CIKM AnalytiCup – 阿里小蜜机器人跨语言短文本匹配算法竞赛](#1-2018cikm-analyticup--阿里小蜜机器人跨语言短文本匹配算法竞赛)
    - [其他](#其他)
  - [二、对话](#二对话)
    - [1. 2020 CCF BDCI《千言：多技能对话》](#1-2020-ccf-bdci千言多技能对话)
      - [1.1 赛题简介](#11-赛题简介)
      - [1.2 比赛方案收集](#12-比赛方案收集)
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
    - [7.1. CCKS&2019中文短文本的实体链指](#71-ccks2019中文短文本的实体链指)
    - [7.2. CCKS 2020实体链指比赛](#72-ccks-2020实体链指比赛)
  - [八、命名实体识别](#八命名实体识别)
    - [8.1 天池中药说明书实体识别](#81-天池中药说明书实体识别)
      - [8.1.1 任务描述](#811-任务描述)
      - [8.1.2 比赛方案](#812-比赛方案)
    - [8.2 CCF BDCI 中文命名实体识别算法鲁棒性评测](#82-ccf-bdci-中文命名实体识别算法鲁棒性评测)
      - [8.2.1 任务描述](#821-任务描述)
  - [九、事件抽取](#九事件抽取)
    - [9.1 CCKS 2020：面向金融领域的小样本跨类迁移事件抽取](#91-ccks-2020面向金融领域的小样本跨类迁移事件抽取)
      - [9.1.1 任务描述](#911-任务描述)
      - [9.1.2 比赛方案](#912-比赛方案)
    - [9.2 CCKS2019_EventEntityExtraction](#92-ccks2019_evententityextraction)
      - [9.1.1 任务描述](#911-任务描述-1)
      - [9.1.2 比赛方案](#912-比赛方案-1)
    - [9.3 2020 科大讯飞事件抽取挑战赛](#93-2020-科大讯飞事件抽取挑战赛)
      - [9.3.1 任务描述](#931-任务描述)
      - [9.3.2 比赛方案](#932-比赛方案)
  - [十、阅读理解](#十阅读理解)
    - [10.1 2021海华AI挑战赛·中文阅读理解·技术组](#101-2021海华ai挑战赛中文阅读理解技术组)
      - [10.1.1 赛题背景](#1011-赛题背景)
      - [10.1.2 比赛任务](#1012-比赛任务)
      - [10.1.3 比赛方案](#1013-比赛方案)
  - [十一、关系抽取](#十一关系抽取)
    - [11.1 2020语言与智能技术竞赛：关系抽取任务 【比赛链接】](#111-2020语言与智能技术竞赛关系抽取任务-比赛链接)
      - [11.1.1 赛题背景](#1111-赛题背景)
      - [11.1.2 赛题说明](#1112-赛题说明)
      - [11.1.3 数据集介绍](#1113-数据集介绍)
      - [11.1.4 比赛方案](#1114-比赛方案)
  - [十二、中文文本纠错](#十二中文文本纠错)
    - [12.1 专业赛：自然语言处理技术创新大赛——中文文本纠错比赛](#121-专业赛自然语言处理技术创新大赛中文文本纠错比赛)
      - [12.1.1 竞赛背景](#1211-竞赛背景)
      - [12.1.2 赛题描述](#1212-赛题描述)
      - [12.1.3 模型训练](#1213-模型训练)
      - [12.1.4 训练集数据介绍](#1214-训练集数据介绍)
      - [12.1.5 比赛方案](#1215-比赛方案)
  - [十三、CCF BDCI](#十三ccf-bdci)
    - [13.1 CCF BDCI 智能人机交互自然语言理解](#131-ccf-bdci-智能人机交互自然语言理解)
      - [13.1.1 竞赛背景](#1311-竞赛背景)
      - [13.1.2 数据集介绍](#1312-数据集介绍)
    - [13.2 CCF BDCI 预训练模型知识量度量](#132-ccf-bdci-预训练模型知识量度量)
      - [13.2.1 竞赛背景](#1321-竞赛背景)
      - [13.2.2 数据集介绍](#1322-数据集介绍)
  - [参考资料](#参考资料)

## 一、问答匹配任务

### 5. 新冠疫情相似句对判定大赛 【[比赛地址](https://tianchi.aliyun.com/competition/entrance/231776/introduction)】

#### 5.1 赛题背景 

面对疫情抗击，疫情知识问答应用得到普遍推广。如何通过自然语言技术将问答进行相似分类仍然是一个有价值的问题。如识别患者相似问题，有利于理解患者真正诉求，帮助快速匹配准确答案，提升患者获得感；归纳医生相似答案，有助于分析答案规范性，保证疫情期间问诊规范性，避免误诊。

本次比赛达摩院联合医疗服务机构妙健康发布疫情相似句对判定任务。比赛整理近万条真实语境下疫情相关的肺炎、支原体肺炎、支气管炎、上呼吸道感染、肺结核、哮喘、胸膜炎、肺气肿、感冒、咳血等患者提问句对，要求选手通过自然语言处理技术识别相似的患者问题。本次比赛成果将作为原子能力助力疫情智能问答应用技术精准度提升，探索下一代医疗智能问答技术，具有广泛的技术和公益价值。

#### 5.2 数据集介绍

本次大赛数据包括：脱敏之后的医疗问题数据对和标注数据。医疗问题涉及“肺炎”、“支原体肺炎”、“支气管炎”、“上呼吸道感染”、“肺结核”、“哮喘”、“胸膜炎”、“肺气肿”、“感冒”、“咳血”等10个病种。

数据共包含train.csv、dev.csv、test.csv三个文件，其中给参赛选手的文件包含训练集train.csv和验证集dev.csv，测试集test.csv 对参赛选手不可见。

每一条数据由Id, Category，Query1，Query2，Label构成，分别表示问题编号、类别、问句1、问句2、标签。Label表示问句之间的语义是否相同，若相同，标为1，若不相同，标为0。其中，训练集、验证集Label已知，测试集Label未知。

示例

```s
    编号：0
    类别：肺炎
    问句1：肺部发炎是什么原因引起的？
    问句2：肺部发炎是什么引起的
    标签:1

    编号：1
    类别：肺炎
    问句1：肺部发炎是什么原因引起的？
    问句2：肺部炎症有什么症状
    标签:0
```

#### 5.3 比赛方案收集

<table>
    <tr>
        <td>名次</td><td>分数</td><td>方案介绍</td> <td>github</td>
    </tr>
    <tr>
        <td>第三名</td>
        <td>*</td>
        <td>[方案介绍](https://github.com/huanghuidmml/epidemicTextMatch)</td>
        <td>[github](https://github.com/huanghuidmml/epidemicTextMatch)</td>
    </tr>
    <tr>
        <td>第一名</td>
        <td>*0.72*</td>
        <td>[方案介绍](https://github.com/zzy99/epidemic-sentence-pair)</td>
        <td>[github](https://github.com/zzy99/epidemic-sentence-pair)</td>
    </tr>
</tabel>


### 4. 2021搜狐校园文本匹配算法大赛 【[比赛地址](https://www.biendata.xyz/competition/sohu_2021/)】

#### 4.1 赛题背景

自然语言理解是人工智能的重大难题之一，也是目前智能语音交互和人机对话的核心挑战。

在自然语言理解中，自然语言推理（Nature Language Inference，NLI）被认为是一个非常基础但重要的研究任务。它要求机器去理解自然语言的深层次语义信息，进而做出合理的推理。更具体的推理任务，则是判断句子关系，即对于给定的两个句子，判断它们含义是否一致。

NLI任务作为一种分类任务，可以帮助评价机器的自然语言理解能力。如果一个模型在NLI任务上有良好表现，则可以认为该模型具备一定的自然语言理解能力。

当前，NLI领域虽然发展迅猛，但仍有诸多难点和挑战，比如如何将常识知识融入机器推理模型中、如何更准确地理解语义等等。

在上述背景之下，搜狐发起并主办 “2021搜狐校园算法大赛”，有针对性地设置30万条数据，总奖金池6.5万元。旨在通过提供业务场景、真实数据、专家指导，选拔和培养有志于自然语言处理领域的算法研究、应用探索的青年才俊，共同探索更多可能、开启无限未来。

#### 4.2 比赛任务

本次比赛的数据均来自人工标注，数据均为文字片段，每两个片段为一组，参赛选手需要为每对文本在两个颗粒度上判断文本对中的两段文字是否匹配。其中，一个颗粒度较为宽泛，两段文字属于一个话题即可视为匹配；另一个颗粒度较为严格，两段文字必须是同一事件才视为匹配。

本次比赛训练数据将分为三个阶段逐步更新，后一阶段任务可以使用前一阶段更新的训练数据。

#### 4.3 数据说明 【[地址](https://www.biendata.xyz/competition/sohu_2021/data/)】

本次比赛初赛预计涉及18万条数据，复赛预计涉及3万条数据，决赛预计涉及9万条数据，数据格式为json。

参赛选手需要正确判断两段文字是否匹配，数据分为A和B两个文件，A和B文件匹配标准不一样。其中，A文件匹配标准较为宽泛，两段文字是同一个话题便视为匹配，B文件匹配标准较为严格，两段文字须是同一个事件才视为匹配。

文件内，source为第一段文字 ，target第二段文字 ，labelA为A文件中匹配情况，labelB为B文件中匹配情况，“0”表示不匹配，“1”表示匹配。

一条样本数据：

- A.txt：

```s
{
    "source": "英国伦敦，20/21赛季英超第20轮，托特纳姆热刺VS利物浦。热刺本赛季18轮联赛是9胜6平3负，目前积33分排名联赛第5位。利物浦本赛季19轮联赛是9胜7平3负，目前积34分排名联赛第4位。从目前的走势来看，本场比赛从热刺的角度来讲，是非常被动的。最终，本场比赛的比分为托特纳姆热刺1-3利",
    "target": " 北京时间1月29日凌晨4时，英超联赛第20轮迎来一场强强对话，热刺坐镇主场迎战利物浦。  热刺vs利物浦，比赛看点如下： 第一：热刺能否成功复仇？双方首回合，热刺客场1-2被利物浦绝杀，赛后穆里尼奥称最好的球队输了，本轮热刺主场迎战利物浦，借着红军5轮不胜的低迷状态，能否成功复仇？ 第二：利物浦近",
    "labelA": "1"
}
```

- B.txt：

```s
{
    "source": "英国伦敦，20/21赛季英超第20轮，托特纳姆热刺VS利物浦。热刺本赛季18轮联赛是9胜6平3负，目前积33分排名联赛第5位。利物浦本赛季19轮联赛是9胜7平3负，目前积34分排名联赛第4位。从目前的走势来看，本场比赛从热刺的角度来讲，是非常被动的。最终，本场比赛的比分为托特纳姆热刺1-3利",
    "target": " 北京时间1月29日凌晨4时，英超联赛第20轮迎来一场强强对话，热刺坐镇主场迎战利物浦。  热刺vs利物浦，比赛看点如下： 第一：热刺能否成功复仇？双方首回合，热刺客场1-2被利物浦绝杀，赛后穆里尼奥称最好的球队输了，本轮热刺主场迎战利物浦，借着红军5轮不胜的低迷状态，能否成功复仇？ 第二：利物浦近",
    "labelB": "0"
}
```

#### 4.4 比赛方案收集

<table>
    <tr>
        <td>名次</td>
        <td>分数</td>
        <td>方案介绍</td>
        <td>github</td>
    </tr>
    <tr>
        <td>*</td>
        <td>baseline 1</td>
        <td>[方案介绍](https://xv44586.github.io/2021/01/20/ccf-qa-2/)</td>
        <td>[github](https://github.com/xv44586/ccf_2020_qa_match)</td>
    </tr>
    <tr>
        <td>baseline 2</td>
        <td>0.72</td>
        <td>[方案介绍](https://www.biendata.xyz/forum/view_post_category/1249/)</td>
        <td>[github](https://github.com/AndyandViky/NLP-Programs/tree/master/03_NLI)</td>
    </tr>
    <tr>
        <td>baseline 3</td>
        <td>0.73</td>
        <td>[方案介绍](https://kexue.fm/archives/8337)</td>
        <td>[github](https://github.com/bojone/sohu2021-baseline)</td>
    </tr>
</tabel>

### 3. CCF2020问答匹配比赛

#### 3.1 比赛背景

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

#### 3.2 比赛方案收集



### 2. 智能客服问题相似度算法设计——第三届魔镜杯大赛

- rank6 https://github.com/qrfaction/paipaidai
- rank12 https://www.jianshu.com/p/827dd447daf9 https://github.com/LittletreeZou/Question-Pairs-Matching
- Rank16：https://github.com/guoday/PaiPaiDai2018_rank16
- Rank29: https://github.com/wangjiaxin24/daguan_NLP

### 1. 2018CIKM AnalytiCup – 阿里小蜜机器人跨语言短文本匹配算法竞赛

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

### 7.1. CCKS&2019中文短文本的实体链指

- [CCKS&百度 2019中文短文本的实体链指 第一名解决方案](https://github.com/panchunguang/ccks_baidu_entity_link)
- [CCKS2019中文短文本实体链指比赛技术创新奖解决方案](https://zhuanlan.zhihu.com/p/403339475)
  - [github 地址](https://github.com/AlexYangLi/ccks2019_el)
- [多因子融合的实体识别与链指消歧](https://zhuanlan.zhihu.com/p/79389393)
- [论文导读 | OpenAI的实体消歧新发现](https://juejin.im/post/6844903566860091399)
- [CCKS&百度 2019中文短文本的实体链指 第一名解决方案](https://github.com/panchunguang/ccks_baidu_entity_link)

### 7.2. CCKS 2020实体链指比赛

- [小米知识图谱团队CCKS 2020实体链指比赛冠军方案](https://zhuanlan.zhihu.com/p/403317178)


## 八、命名实体识别

### 8.1 天池中药说明书实体识别

#### 8.1.1 任务描述

人工智能加速了中医药领域的传承创新发展，其中中医药文本的信息抽取部分是构建中医药知识图谱的核心部分，为上层应用如临床辅助诊疗系统的构建（CDSS）等奠定了基础。本次NER挑战需要抽取中药药品说明书中的关键信息，包括药品、药物成分、疾病、症状、证候等13类实体，构建中医药药品知识库。

#### 8.1.2 比赛方案

<table>
    <tr>
        <td>名次</td>
        <td>分数</td>
        <td>方案介绍</td>
        <td>github</td>
    </tr>
    <tr>
        <td>1</td>
        <td>72.90% </td>
        <td>[方案介绍](https://github.com/z814081807/DeepNER)</td>
        <td>[github](https://github.com/z814081807/DeepNER)</td>
    </tr>
</tabel>

### 8.2 [CCF BDCI 中文命名实体识别算法鲁棒性评测](https://www.datafountain.cn/competitions/510)

#### 8.2.1 任务描述

赛题名称：中文命名实体识别算法鲁棒性评测

赛题背景：信息检索旨在满足人类在互联网上快速准确地获取信息与知识的需求，具体来说，信息检索是指从大规模非结构化数据中找出满足用户需求的资料的过程。其中，命名实体识别这类文本挖掘基础算法，在非结构化文本转化为结构化文本的过程中起到了至关重要的作用，为信息检索任务性能的提升提供了强有力支撑。然而，尽管深度学习模型在现有数据集中有着出色的表现，在现实应用中通常会出现很大的性能下降。其中一个很重要的原因是缺乏对模型鲁棒性的重视和验证，导致模型只能在特定语料中圈地为王，在现实应用中的性能却大打折扣。

鲁棒性是机器学习模型的一项重要评价指标，主要用于检验模型在面对输入数据的微小变动时，是否依然能保持判断的准确性，也即模型面对一定变化时的表现是否稳定。鲁棒性的高低直接决定了机器学习模型的泛化能力。在现实世界的应用场景中，模型要面对的是更加纷繁复杂的语言应用方式，待处理的数据里包含着更加庞杂的变化。一旦缺乏鲁棒性，模型在现实应用中的性能就会大打折扣。在测试数据集上获得高分是远远不够的，机器学习模型的设计目标是让模型在面对新的外部数据时依然维持精准的判断。因此，为了确保模型的实际应用价值，对模型进行鲁棒性评测是不可或缺的。

近期，复旦大学自然语言处理实验室发布了模型鲁棒性评测平台 TextFlint（textflint.io），该平台涵盖 12 项 NLP 任务，囊括 80 余种数据变形方法，涵盖了领域相关黑盒变形、领域无关黑盒变形、白盒变形、分组抽样、分析报告等等一系列功能，为研究人员提供一个便捷的模型鲁棒性验证方法。为了构建安全可靠的深度学习系统，消除深度学习模型在实际部署应用中的潜在鲁棒性风险，本赛题针对命名实体任务开展鲁棒性评测。本次评测提供给参赛者的原始数据集训练模型，而验证集和测试集由鲁棒性评测工具 TextFlint 产生，以最终排名依据为新的鲁棒性验证集和测试集的性能表现。

赛题任务：给定未分好词但有实体标记的中文文本训练数据，设计算法对验证集中词语的实体类型进行预测，其中部分文本经过了鲁棒性评测工具TextFlint的扰动。预测的任务是判断哪些词语是人名(PER)，地理位置(LOC)，机构(ORG)或行政单位(GPE)，并以(B,M,E)的格式给出词语的边界。

## 九、事件抽取

### 9.1 CCKS 2020：面向金融领域的小样本跨类迁移事件抽取

#### 9.1.1 任务描述

在金融领域，事件抽取是一项十分重要的任务，也是自然语言处理领域一项比较复杂的任务，而小样本下的事件抽取模型在落地应用中也极为需要。本任务需要从金融领域新闻资讯句子中，抽取事件知识（包括事件类型、触发词和事件元素），并将大样本下训练的模型跨类迁移到小样本的其他事件类型上。

其中，事件类型分为两类，初始事件类型限定为：质押、股份股权转让、投资、起诉和减持，需要迁移的事件类型为：收购、担保、中标、签署合同和判决，每个事件类型都有其对应的事件框架，需要抽取出每个事件对应的事件元素 。即给出一段句子级新闻资讯文本，针对该文本需要判断其所属的事件类型，抽取该事件的各个事件元素。

#### 9.1.2 比赛方案

<table>
    <tr>
        <td>名次</td>
        <td>分数</td>
        <td>方案介绍</td>
        <td>github</td>
    </tr>
    <tr>
        <td>——</td>
        <td>—— </td>
        <td>[方案介绍](https://github.com/xiaoqian19940510/CCKS-2020-event-extraction)</td>
        <td>[github](https://github.com/xiaoqian19940510/CCKS-2020-event-extraction)</td>
    </tr>
</tabel>


### 9.2 CCKS2019_EventEntityExtraction

#### 9.1.1 任务描述

“事件识别”是舆情监控领域和金融领域的重要任务之一，“事件”在金融领域是投资分析，资产管理的重要决策参考。“事件识别”的复杂性在于事件类型和事件主体的判断，比如“公司A产品出现添加剂，其下属子公司B和公司C遭到了调查”，对于“产品出现问题”事件类型，该句中事件主体是“公司A”，而不是“公司B”或“公司C”。我们称发生特定事件类型的主体成为事件主体，本任务中事件主体范围限定为：公司和机构。事件类型范围确定为：产品出现问题、高管减持、违法违规…

本次评测任务的主要目标是从真实的新闻语料中，抽取特定事件类型的主体。即给定一段文本T，和文本所属的事件类型S，从文本T中抽取指定事件类型S的事件主体。

输入：一段文本，事件类型S

输出：事件主体

示例：

样例1

输入：”公司A产品出现添加剂，其下属子公司B和公司C遭到了调查”， “产品出现问题”

输出： “公司A”

样例2

输入：“公司A高管涉嫌违规减持”，“交易违规”

输出： “公司A”

#### 9.1.2 比赛方案

<table>
    <tr>
        <td>名次</td>
        <td>分数</td>
        <td>方案介绍</td>
        <td>github</td>
    </tr>
    <tr>
        <td>5</td>
        <td>—— </td>
        <td>[方案介绍](https://github.com/hecongqing/CCKS2019_EventEntityExtraction_Rank5)</td>
        <td>[github](https://github.com/hecongqing/CCKS2019_EventEntityExtraction_Rank5)</td>
    </tr>
</tabel>

### 9.3 [2020 科大讯飞事件抽取挑战赛](http://challenge.xfyun.cn/topic/info?type=hotspot)

#### 9.3.1 任务描述

本赛事任务旨在从通用新闻文本中抽取事件触发词、事件论元以及事件属性。 在传统的事件定义中，事件由事件触发词( Trigger) 和描述事件结构的元素 (Argument)构成。事件触发词标识着事件的发生。事件论元为事件主体(Subject)、客体(Object)、时间(Time)、地点(Location)等，是表达事件重要信息的载体。

事件属性包括事件极性（Polarity）、时态(Tense)，是衡量事件是否真实发生的重要依据。 通过极性，事件分为肯定、否定、可能事件。通过时态，事件分为过去发生的事件、现在正 在发生的事件、将要发生的事件以及其他无法确定时态的事件。

本赛事任务一为初赛任务，任务二为复赛任务，在任务一的基础上增加了事件属性识别。为 了模拟真实场景，数据中包含了非实际发生的事件。

任务一：事件触发词及论元抽取

该任务旨在从文本中抽取标识事件发生的触发词和论元，触发词往往为动词和名词。触发词 对应的事件论元，主要为主体、客体、时间、地点，其中主体为必备论元。


任务二：事件属性抽取

该任务旨在从文本中抽取表达事件发生状态的属性，包括极性、时态。极性分为：肯定、否定、可能；时态分为：过去、现在、将来、其他。

#### 9.3.2 比赛方案

<table>
    <tr>
        <td>名次</td>
        <td>分数</td>
        <td>方案介绍</td>
        <td>github</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.73859</td>
        <td>[方案介绍](https://github.com/km1994/xf_event_extraction2020Top1)</td>
        <td>[github](https://github.com/km1994/xf_event_extraction2020Top1)</td>
    </tr>
</tabel>

## 十、阅读理解

### 10.1 2021海华AI挑战赛·中文阅读理解·技术组

#### 10.1.1 赛题背景

文字是人类用以记录和表达的最基本工具，也是信息传播的重要媒介。透过文字与符号，我们可以追寻人类文明的起源，可以传播知识与经验，读懂文字是认识与了解的第一步。对于人工智能而言，它的核心问题之一就是认知，而认知的核心则是语义理解。
 
机器阅读理解(Machine Reading Comprehension)是自然语言处理和人工智能领域的前沿课题，对于使机器拥有认知能力、提升机器智能水平具有重要价值，拥有广阔的应用前景。机器的阅读理解是让机器阅读文本，然后回答与阅读内容相关的问题，体现的是人工智能对文本信息获取、理解和挖掘的能力，在对话、搜索、问答、同声传译等领域，机器阅读理解可以产生的现实价值正在日益凸显，长远的目标则是能够为各行各业提供解决方案。
 
《2021海华AI挑战赛·中文阅读理解》大赛由中关村海华信息技术前沿研究院与清华大学交叉信息研究院联合主办，腾讯云计算协办。共设置题库16000条数据，总奖金池30万元，且腾讯云计算为中学组赛道提供独家算力资源支持。
 
本次比赛的数据来自小学/中高考语文阅读理解题库（其中，技术组的数据主要为中高考语文试题，中学组的数据主要来自小学语文试题）。相较于英文，中文阅读理解有着更多的歧义性和多义性，然而璀璨的中华文明得以绵延数千年，离不开每一个时代里努力钻研、坚守传承的人，这也正是本次大赛的魅力与挑战，让机器读懂文字，让机器学习文明。秉承着人才培养的初心，我们继续保留针对中学组以及技术组的两条平行赛道，科技创新，时代有我，期待你们的回响。
 
#### 10.1.2 比赛任务

本次比赛技术组的数据来自中高考语文阅读理解题库。每条数据都包括一篇文章，至少一个问题和多个候选选项。参赛选手需要搭建模型，从候选选项中选出正确的一个。

#### 10.1.3 比赛方案

<table>
    <tr>
        <td>名次</td>
        <td>分数</td>
        <td>方案介绍</td>
        <td>github</td>
    </tr>
    <tr>
        <td>torch版baseline</td>
        <td>35.88380</td>
        <td>[方案介绍](https://www.biendata.xyz/forum/view_post_category/1211/)</td>
        <td>[github](https://github.com/jiangxiaoshuaiya/BertHub)</td>
    </tr>
    <tr>
        <td>bert 处理超长文本一个baseline</td>
        <td>*</td>
        <td>[方案介绍](https://www.biendata.xyz/forum/view_post_category/1216/)</td>
        <td>[github](https://github.com/hackerxiaobai/haihua_QA)</td>
    </tr>
</tabel>

## 十一、关系抽取

### 11.1 2020语言与智能技术竞赛：关系抽取任务 【[比赛链接](https://aistudio.baidu.com/aistudio/competition/detail/31?isFromCcf=true)】

#### 11.1.1 赛题背景

关系抽取 (Relation Extraction, RE) 是从自然语言文本中抽取实体及其之间关系的信息技术，是信息检索、智能问答、智能对话等人工智能应用的重要基础，一直受到业界的广泛关注。关系抽取任务涉及命名实体识别、指代消解、关系分类等复杂技术，极具挑战性。
RE (Relation Extraction) is a technology of automatically extracting entities and their relationships from textual data. It is an important foundation for artificial intelligence applications such as information retrieval, intelligent question answering, and intelligent dialogue, and has been widely concerned by the industry. RE is an extremely challenging job since it involves several difficult tasks such as named entity recognition, coreference resolution and relation classification.

本次竞赛在去年信息抽取竞赛的基础上进行了升级。整体的任务形式仍然是 schema 约束下的关系抽取，也就是在给定关系集合下，从自然语言文本中抽取出符合关系 schema 约束的 SPO 三元组知识，但是对 O 值形态进行了复杂化的扩展。相信这会给参赛者带来更大的挑战和乐趣。
The RE competition of this year has been upgraded on the basis of IE (Information Extraction) competition of last year. The major focus is still extracting the SPO triples from texts with the constraint of given schemas, but extending to the phase where now the O (object) can have multiple slots and values. This upgrade will bring more fun and challenge to our contestants.

除此之外，本次竞赛将提供业界规模最大的中文关系抽取数据集 DuIE 2.0，旨在为研究者提供学术交流平台，进一步提升中文关系抽取技术的研究水平，推动相关人工智能应用的发展。
Besides, this year the competition will provide the industry’s largest Chinese Relation Extraction dataset DuIE 2.0, and hopefully, it would be an academic platform for researchers to exchange ideas and could help in promoting the development of RE technologies and applications.

#### 11.1.2 赛题说明

任务目标是在给定的文本句子中，根据预先定义的schema集合，抽取出所有满足 schema 约束的 SPO 三元组。schema 定义了关系 P 以及其对应的主体 S 和客体 O 的类别，根据 O 类型的复杂程度可以划分为以下两种：
The target of the task is to extract and output all correct SPO triples from a given sentence sent that are constrained by the pre-defined schema sets provided. Schema defines the relation P and its corresponding classes of subject S and object O. Based on the complexity of the object O, there are two types of schemas:

1.简单 O 值：也就是说 O 是一个单一的文本。例如，「妻子」关系的 schema 定义为：
Single-O-value schema (Schema with single object slot and value), which means in this type of schema the object O contains only one single slot and value. For example, a schema with P of value [wife] can be defined as:

```s
{
    S_TYPE: 人物,
    P: 妻子,
    O_TYPE: {
        @value: 人物
	}
}
```

简单 O 值是最常见关系类型，去年竞赛中所发布的所有 schema 都属于这种类型。为了保持格式统一，简单 O 值类型的 schema 定义通过结构体保存，结构体中只有一个 @value 字段存放真正的 O 值类型。
Single-O-value schema is the most common type of schema. All schemas released last year fall into this category. In order to maintain a consistent formatting style, all definition of single-O-value schemas are stored in accordance with the format of a struct. In single-O-value schema, there can be only one slot (@value ) for storing the value (person) of its corresponding object O.

2.复杂 O 值：也就是说 O 是一个结构体，由多个语义明确的文本共同组成，多个文本对应了结构体中的多个槽位（slot）。例如，「饰演」关系中 O 值有两个槽位 @value 和 inWork，分别表示「饰演的角色是什么」以及「在哪部影视作品中发生的饰演关系」，其 schema 定义为：
Multiple-O-values schema (Schema with multiple object slots and values) where each of the multiple slots of the schema is filled by one corresponding value made up by texts of clear definition. For instance, in a schema with P of value [play], there can be two slots [@value] and [inWork] representing respectively [what is the role the person is playing] and [in what film or television work the person plays the role]. It can be defined in the following way:

```s
{
    S_TYPE: 娱乐人物,
    P: 饰演,
    O_TYPE: {
        @value: 角色
        inWork: 影视作品
    }
} 
```

在复杂 O 值类型的定义中，@value 槽位可以认为是该关系的默认 O 值槽位，对于该关系不可或缺，其他槽位均可缺省。
In the definition of multiple-O-values schema, there is supposed to be a primary or default object slot in the multiple-O-values schema where in this case is the [@value]. This default slot of the object is indispensable while other slots are not.

```s
输入/输出：
Input/Output:

输入：schema约束集合及句子sent
Input : A list of pre-defined schemas and sentence sent.

输出： 句子sent中包含的符合给定schema约束的三元组知识Triples
Output : Triples mentioned in sent under the constraints of given schemas.
```

#### 11.1.3 数据集介绍

本次竞赛数据集共包含 48个已定义好的schema和超过21万中文句子，其中包括17万训练集，2万验证集和2万测试集，共分为以下5个部分：
The dataset includes a list of 48 pre-defined common schemas, a training set (170k sentences), a development set (20k sentences) and 2 test sets (20k sentences), containing 210k sentences total, and can be downloaded in 5 parts.

1. 训练集：共17万个句子，包含句子中对应的SPO，用于竞赛模型训练。
Training set: 170k sentences in total, including the corresponding SPO in the sentences for model training.

2. 验证集：共2万个句子，包含句子中对应的SPO，用于竞赛模型训练和参数调试。
Development set: 20k sentences in total, including the corresponding SPO in the sentences for model training and parameter tuning.

3. schema约束：共48个限定的schema，定义了关系P以及其对应的主体S和客体O的类别。
Schema: defines the relation P and its corresponding classes of subject S and object O.

4. 测试集1：约1万个句子，不包含句子中对应的SPO，用于参赛者在平台上自助提交模型预测结果、验证效果。
Test set 1: about 10000 sentences without SPO triples for model verification. It is used for participants to submit the prediction result and verify model on the platform.

5. 测试集2:本次竞赛最终测试集，约2万个句子，不包含句子对应的SPO，包含测试集1。另外为了防止针对测试集的调试，数据中将会额外加入混淆数据。该部分数据在评测结束前一周发布，结果不能在平台上自助验证，由评测委员会进行离线评测。
Test set 2: the final test data of the challenge, about 20000 sentences in total, contains the content of Test set 1 but does not annotate SPO triples. Noises are added to rule out tuning against the test set. The results of test set 2 is not available for online model verification.

样例数据 Data Sample
平台提供的数据为JSON文件格式，样例如下:
The data is provided in JSON format as follows:

```s
{
    "text":"王雪纯是87版《红楼梦》中晴雯的配音者，她是中央台《正大综艺》的主持人",
    "spo_list":[
        {
            "predicate":"配音",
            "subject_type":"娱乐人物",
            "object":{
                "@value":"晴雯",
                "inWork":"红楼梦"
            },
            "object_type":{
                "@value":"人物",
                "inWork":"影视作品"
            },
            "subject":"王雪纯"
        },
        {
            "predicate":"主持人",
            "subject_type":"电视综艺",
            "object":{
                "@value":"王雪纯"
            },
            "object_type":{
                "@value":"人物"
            },
            "subject":"正大综艺"
        }
    ]
}
```

#### 11.1.4 比赛方案

<table>
    <tr>
        <td>名次</td>
        <td>分数</td>
        <td>方案介绍</td>
        <td>github</td>
    </tr>
    <tr>
        <td>15/1392</td>
        <td>0.7695</td>
        <td>[方案介绍](https://github.com/aker218/Baidu-2020-Language-and-Intelligent-Technology-Competition-Relation-Extraction-rank15)</td>
        <td>[github](https://github.com/aker218/Baidu-2020-Language-and-Intelligent-Technology-Competition-Relation-Extraction-rank15)</td>
    </tr>
    <tr>
        <td>Relation-Extraction-2020</td>
        <td>*</td>
        <td>[方案介绍](https://github.com/Karl8/Relation-Extraction-2020)</td>
        <td>[github](https://github.com/Karl8/Relation-Extraction-2020)</td>
    </tr>
</tabel>

## 十二、中文文本纠错

### 12.1 [专业赛：自然语言处理技术创新大赛——中文文本纠错比赛](http://2021aichina.caai.cn/track?id=5)

#### 12.1.1 竞赛背景

文本校对任务主要是针对文本中出现的错误进行检测和纠正，属于综合性的自然语言处理研究子方向，能够比较全面体现了自然语言处理的技术水平。过往文本校对相关评测使用的都是外国语言学习者撰写的文本，这些文本的错误大多数都是一些中文母语写作者不会犯的一些错误。对于政务公文、新闻出版等行业来说，一款针对以中文为母语的用户所使用的校对系统将会有更大的帮助。因此，本赛题主要选择互联网上中文母语写作者撰写的网络文本作为校对评测数据，从拼写错误、语法错误、语病错误等多个方面考察机器的认知智能能力。

#### 12.1.2 赛题描述

本次赛题选择网络文本作为校对数据，从中检测并纠正错误，实现中文文本校对系统。即给定一段文本，校对系统从中检测出错误字词、错误类型，并进行纠正。具体的输入、输出及错误类型为：

```s
    输入：输入文件包含若干行文本，每行文本对应句子ID和相应的待校对句。

    输出：输出文件每行对应句子ID及相应的校对结果。校对结果中每处错误需包含错误位置、错误类型、错误字词及正确字词，每处错误及多处错误间均以英文逗号分隔，文件编码采用utf8编码。
```

> 错误类型：拼写错误，包括别字及别词；语法错误，包括冗余、缺失、乱序；语病错误，包括语义重复及句式杂糅。

- 输入文件示例：

```s
    pid=0011-1 关于瑞典时装公司HM拒绝使用新疆产品的言轮在华引发广泛声讨和抵制浪潮，有记者就此提问。华春莹标识：
    pid=0011-2 新疆棉花是世界上最好的棉花之一，不用是相关企业的损失；
    pid=0011-3 给老百姓包括少数民族群众提供更多的就业机会，一般正常人都都会觉得是件好事。
    pid=0011-4 因为他们自己上历史真的就这么干了上百年，所以现在以己度人；
    pid=0012-1 第三局比赛俄罗女排的气势被完全压制，中国女排就此以一场3比0零封取得五连胜。
    pid=0013-1 ④中国光明磊落，中国人民友善开放，但中国民意不可欺不可违。
    pid=0023-1 对学校的未来发展，专家们提出了许多真知灼见的意见。
    pid=0069-1 高速公路上交通事故的主要原因是司机违反交通规则或操作不当造成的。
```

- 输出文件示例：

```s
    pid=0011-1, 20, 别字, 轮, 论, 46, 别词, 标识, 表示,
    pid=0011-2, -1
    pid=0011-3, 28, 冗余, 都, ,
    pid=0011-4, 6, 乱序, 上历史, 历史上,
    pid=0012-1, 7, 缺失, , 斯,
    pid=0013-1, -1
    pid=0023-1, 21, 语义重复, 的意见, ,
    pid=0069-1, 29, 句式杂糅, 造成的, ,
```

#### 12.1.3 模型训练

主办方提供如下解题思路供参赛选手参考，选手可按照如下思路开发文本校对模型，也可选择其他人工智能方法，思路不限。


- 解题思路：

1. 使用序列标注模型进行错误检测，使用语言模型对错误进行纠正
2. 使用Seq2seq模型对整个句子进行纠正

#### 12.1.4 训练集数据介绍

主办方提供部分伪数据，初赛大致规模为10万句，复赛将公开更多伪数据。参赛队伍也可以使用公开数据集及其他开放型数据。

- 训练数据及Baseline将在如下网站进行公开：https://github.com/destwang/CTC2021
- 现有公开数据集如下：
  - CGED历年公开数据集 http://www.cged.tech/
  - NLPCC2018语法纠错数据集 http://tcci.ccf.org.cn/conference/2018/taskdata.php
  - SIGHAN及相关训练集 http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html

#### 12.1.5 比赛方案

<table>
    <tr>
        <td>名次</td>
        <td>分数</td>
        <td>方案介绍</td>
        <td>github</td>
    </tr>
    <tr>
        <td>baseline</td>
        <td>baseline</td>
        <td>[CIEC-CTC 2021 (Chinese Text Correction)](https://github.com/destwang/CTC2021)</td>
        <td>[CIEC-CTC 2021 (Chinese Text Correction)](https://github.com/destwang/CTC2021)</td>
    </tr>
</tabel>

## 十三、CCF BDCI

### 13.1 [CCF BDCI 智能人机交互自然语言理解](https://www.datafountain.cn/competitions/511)

#### 13.1.1 竞赛背景

赛题名称：智能人机交互自然语言理解

赛题背景：人机交互是人工智能的重要应用领域，随着人工智能技术的发展，越来越多的智能交互产品出现在人们的日常生活中。在交互方式上，“手势交互”，“语音交互”，“AR交互”等新兴交互方式开始出现在公众的视野中，但由于交互效率和人体工学等方面的限制，手势交互等方法短期内较难成为主流的人机交互方式，而搭载了语音交互能力的产品自落地应用起就一直受到极为广泛的关注。在交互手段方面，用户仅需要通过与相关产品对话即可下达命令、完成播放音乐、控制家居等任务，能够真正意义上解放双手，提升生活幸福指数。但在实际应用中，相关产品往往很难满足用户的各类别复杂要求，其根源在于自然语言本身较高的复杂性使得用户意图无法被较好的理解。自然语言理解（NLU ,Natural Language Understanding）任务旨在让计算机具备理解用户语言的能力，从而进行下一步决策或完成交互动作，是具备语音交互能力的产品需要处理的重要任务。

在产品的研发迭代过程中，为满足用户的各类使用需求，往往需要大量标注数据用于训练才能达成较好的交互效果，而大量的标注数据意味着高昂的成本投入。小样本学习（Few-shot Learning）是机器学习领域近年来新兴的研究方向之一，旨在凭借基础类别中学到的先验知识及少量新类别标记样本，较好的完成对新类别数据的泛化任务，从而减少数据成本投入的需求。另一方面，产品在实际应用中可能经常遇到未经训练的新类别问题，如何识别并处理此种域外问题同样是生产中的常见问题之一。域外检测（Out-of-Distribution Detection）任务旨在帮助产品较好的处理真实使用场景中的未知意图，挖掘出用户的潜在意图，从而帮助提升服务质量。

为了更好的解决智能人机交互产品的自然语言理解任务，本赛题对NLU领域的意图识别及槽位填充任务进行考察。除基本的学习任务外，还期望通过小样本学习任务减少产品对大量新类别标注数据的依赖，通过域外检测任务识别未知意图，摆脱对已知意图的干扰，同时达到尽可能好的学习效果。

赛题任务：根据用户与系统的单轮对话，识别对话用户意图并进行槽位填充。除基础的意图识别及槽位填充任务外，本赛题额外包括2个子任务：
人机交互-NLU-1（小样本学习任务）：根据基础意图类别数据及少量含标注的新意图类别样本，完成新意图类别的识别及槽位填充任务。
人机交互-NLU-2（域外意图检测任务）：除识别出训练数据中已知的意图类别外，对于未知意图类别数据进行检测。
注：入围团队需提供代码、说明文件等材料供审查。

#### 13.1.2 数据集介绍

- 数据简介

数据集包含用户与音箱等智能设备进行单轮对话的文本数据，包含共11个意图类别，共计约47个槽位类型。
注：本赛题允许使用外部数据集。

- 数据说明

竞赛数据以json格式存储，包括train.json及test.json两个文件。

train.json为训练数据，共约9100条，其文本由ASR语音识别技术转写得到，因此可能需要进一步处理，文本标注包含用户意图、槽位及槽值三个字段，具体数据构成如下：

1. 基础任务数据：包含9个意图类别，共约9000条，各类别数据分布均衡。
2. 小样本任务数据：包含2个意图类别，每个意图类别含标注数据约50条，且任意一个意图类别和槽位至少被5条语料覆盖。
3. 域外检测任务数据：不提供训练数据，将以上11个类别之外的数据识别为“Other”意图类别即可，不需进行槽位填充。

test.json为测试数据，A、B榜数据各3000条，共约6000条。



### 13.2 [CCF BDCI 预训练模型知识量度量](https://www.datafountain.cn/competitions/509)

#### 13.2.1 竞赛背景

赛题名称：预训练模型知识量度量

赛题背景：近年来，以BERT为代表的预训练模型在各项NLP任务中取得了重大突破。相关研究表明，预训练模型不仅可以学习通用语言表示，还可以学习结构化的知识，包括常识知识和事实知识。以BERT为例，可以通过完形填空任务记忆结构化的事实知识，如“Einstein was born in the country [Mask]”即在捕捉（Einstein, born-in, Germany）的三元组信息。模型将知识编码进上亿参数中，使一系列的下游任务，如常识（视觉）问题解答、信息检索，可以从预先捕捉的事实知识和常识知识中受益。

为了更深入地理解预训练模型，我们构建数据集来系统地评估模型的知识含量，不仅考察模型预训练阶段所编码的知识量，同时考察模型是否具备推理能力。为此，我们构造了来自不同领域（历史、军事、医学等）、不同类型（事实知识、常识知识）、不同难度（单条知识、多条知识组合）的问题，把对模型知识量的测评，转换为相应的完型填空问题。当模型能成功填充[MASK]标识符时，则说明该模型掌握了相关知识。

我们希望通过考察模型的知识量，驱动业界对于模型知识表示的研究，实现预训练模型向下游任务更有效的知识转移。 本次预训练模型的知识量评估侧重于考察模型的记忆能力，特别是在few-shot或zero-shot情况下的记忆和能力和多知识组合能力，后期会加入更多的推理类问题，以及考察模型本身的泛化能力。

赛题任务：本赛题构建了完型填空形式的英文测评数据集，评估预训练模型在9个领域、两大知识类型（事实知识、常识知识）、不同难度任务上的知识含量。

#### 13.2.2 数据集介绍

在训练集中，每行数据即为一个问题，包含5个字段

> qid：问题ID <br/>
> query：待填充的句子，[MASK]标识符代表要填充的位置；<br/>
> answer: [MASK]处正确答案(不定长的字符串)的列表（可能不止一个正确答案）<br/>
> domain: 问题类型（Facts代表事实知识，Common Sense代表常识知识）<br/>
> NeedReasoning: False代表简单问题（单条知识）、True代表复杂问题（多条知识组合）<br/>

示例：

![](img/赛题上传1.png)

注：

1. 对于数值类问题，答案格式统一为阿拉伯数字；日期类问题，月份用英文表达，年和日用数字表达，评定与年月日顺序无关；常识类问题，答案限制为单个词语。

![](img/赛题上传2.png)

2. 测试集提供qid和 query。

> qid：问题ID<br/>
> query：待填充的句子，[MASK]标识符代表要填充的位置；



## 参考资料

1. [Data competition Top Solution 数据竞赛Top解决方案开源整理](https://github.com/Smilexuhc/Data-Competition-TopSolution)
2. [nlp-competitions-list-review](https://github.com/zhpmatrix/nlp-competitions-list-review)


