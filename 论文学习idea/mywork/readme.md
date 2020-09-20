# 个人工作总结

> 作者：杨夕
> 
> github：https://github.com/km1994/

## 目录

- [个人工作总结](#个人工作总结)
  - [目录](#目录)
  - [工作一 ： baiduES 面向百度百科的基于知识图谱的搜索引擎构建](#工作一--baidues-面向百度百科的基于知识图谱的搜索引擎构建)
    - [一、项目介绍](#一项目介绍)
    - [二、项目流程](#二项目流程)
    - [三、工程目录](#三工程目录)
    - [四、模块介绍](#四模块介绍)
      - [step 1: 编写 网络爬虫 爬取 百度百科 数据](#step-1-编写-网络爬虫-爬取-百度百科-数据)
      - [step 2: 数据预处理](#step-2-数据预处理)
        - [1）清洗掉 文本中噪声数据](#1清洗掉-文本中噪声数据)
        - [2）根据 标签类型 划分 不同的 类簇](#2根据-标签类型-划分-不同的-类簇)
      - [step 3: 将数据 导入 ES](#step-3-将数据-导入-es)
      - [step 4: 利用 python 编写 后台，并 对 ES 进行查询，返回接口数据](#step-4-利用-python-编写-后台并-对-es-进行查询返回接口数据)
      - [step 5: ES 数据前端展示](#step-5-es-数据前端展示)
      - [step 6：实体消歧，（地点、人物等）](#step-6实体消歧地点人物等)
      - [step 7：实体归一，（地点、人物等）](#step-7实体归一地点人物等)
      - [step 8：各类实体关系 实体库 和 关系库 构建](#step-8各类实体关系-实体库-和-关系库-构建)
      - [step 9：数据 导入 neo4j 图数据库](#step-9数据-导入-neo4j-图数据库)
      - [step 10: 编写 查询 接口，并用于 前台 展示](#step-10-编写-查询-接口并用于-前台-展示)
  - [工作二 ： ChineseEDA 中文 数据增强](#工作二--chineseeda-中文-数据增强)
    - [一、项目介绍](#一项目介绍-1)
    - [二、功能介绍](#二功能介绍)
  - [工作三 ： TextClassifier 中文文本分类任务](#工作三--textclassifier-中文文本分类任务)
    - [一、项目介绍](#一项目介绍-2)
    - [二、requirements](#二requirements)
    - [三、文件目录](#三文件目录)
    - [四、方法介绍](#四方法介绍)
      - [word2vec 词向量预训练](#word2vec-词向量预训练)
        - [介绍](#介绍)
        - [思路介绍](#思路介绍)
      - [fastText 文本分类](#fasttext-文本分类)
        - [介绍](#介绍-1)
        - [思路介绍](#思路介绍-1)
      - [TextCNN 文本分类](#textcnn-文本分类)
        - [介绍](#介绍-2)
        - [思路介绍](#思路介绍-2)
      - [TextRNN 文本分类](#textrnn-文本分类)
        - [介绍](#介绍-3)
        - [思路介绍](#思路介绍-3)
      - [Bi-LSTM + Attention 文本分类](#bi-lstm--attention-文本分类)
        - [介绍](#介绍-4)
        - [思路介绍](#思路介绍-4)
      - [RCNN 文本分类](#rcnn-文本分类)
        - [介绍](#介绍-5)
        - [思路介绍](#思路介绍-5)
      - [Adversarial LSTM 文本分类](#adversarial-lstm-文本分类)
        - [介绍](#介绍-6)
        - [思路介绍](#思路介绍-6)
      - [Transformer  文本分类](#transformer-文本分类)
        - [介绍](#介绍-7)
        - [思路介绍](#思路介绍-7)
  - [工作四 ：text_feature_extraction 文本特征提取](#工作四-text_feature_extraction-文本特征提取)
    - [一、介绍](#一介绍)
    - [二、 TF-IDF关键词提取算法](#二-tf-idf关键词提取算法)
      - [理论基础](#理论基础)
        - [介绍](#介绍-8)
        - [计算公式](#计算公式)
    - [二、PageRank算法](#二pagerank算法)
      - [理论学习](#理论学习)
    - [三、TextRank算法](#三textrank算法)
  - [工作五 NERer 中文命名实体识别](#工作五-nerer-中文命名实体识别)
    - [一、项目介绍](#一项目介绍-3)
    - [二、目录介绍](#二目录介绍)
  - [工作六 TextMatching 中文 文本匹配 方法](#工作六-textmatching-中文-文本匹配-方法)
    - [一、项目介绍](#一项目介绍-4)
    - [二、目录介绍](#二目录介绍-1)
    - [三、模型效果对比](#三模型效果对比)
  - [工作七 textSummarization 中文 文本摘要 方法](#工作七-textsummarization-中文-文本摘要-方法)
    - [一、介绍](#一介绍-1)
    - [二、目录介绍](#二目录介绍-2)
    - [三、 安装环境](#三-安装环境)
    - [四、 示例](#四-示例)
  - [工作八 QAer 中文 问答](#工作八-qaer-中文-问答)
    - [一、 项目介绍](#一-项目介绍)
    - [二、目录介绍](#二目录介绍-3)
    - [三、项目思路介绍](#三项目思路介绍)
  - [工作九 seq2seqAttn 解决不同 的 生成式 任务](#工作九-seq2seqattn-解决不同-的-生成式-任务)
    - [一、介绍](#一介绍-2)
    - [二、目录介绍](#二目录介绍-4)
    - [三、项目进度](#三项目进度)
  - [工作十 rasa 学习](#工作十-rasa-学习)
    - [介绍](#介绍-9)
    - [目录介绍](#目录介绍)


## 工作一 ： baiduES 面向百度百科的基于知识图谱的搜索引擎构建

### 一、项目介绍

该项目为了实现一个 面向百度百科的基于知识图谱的搜索引擎构建。

### 二、项目流程

- step 1: 编写 网络爬虫 爬取 百度百科 数据；
- step 2: 数据预处理：
  - 清洗掉 文本中噪声数据；
  - 根据 标签类型 划分 不同的 类簇（人物、动物、奖项、书籍、公司、事件、食物、游戏、电影、地点、角色、学校、话剧、歌曲、学术用语、物质、词语）
- step 3: 将数据 导入 ES ；
- step 4: 利用 python 编写 后台，并 对 ES 进行查询，返回接口数据；
- step 5: ES 数据前端展示；
- step 6：实体消歧，（地点、人物等）；
- step 7：实体归一，（地点、人物等）；
- step 8：各类实体关系 实体库 和 关系库 构建；
- step 9：数据 导入 neo4j 图数据库；
- step 10: 编写 查询 接口，并用于 前台 展示；

### 三、工程目录

- s1_urllib_study: 网络爬虫 编写模块；
- s2_data_process: 数据预处理模块；
- s3_es_study: 数据导入 ES 模块;
- s4_neo4j_study: neo4j 图数据模块；
- s5_web: 展示界面模块；
- s6_es_search：es 查询；

### 四、模块介绍

#### step 1: 编写 网络爬虫 爬取 百度百科 数据

本模块 通过编写爬虫爬取 百度百科 数据，总共爬取 名称、链接、简介、中文名、外文名、国籍、出生地、出生日期、职业、类型、中文名称、代表作品、民族、主要成就、别名、毕业院校、导演、制片地区、主演、编剧、上映时间 等400多个 指标，共爬取数据。

#### step 2: 数据预处理

##### 1）清洗掉 文本中噪声数据

编写 正则表达式 清洗 数据中的 多余 字符（换行符、制表符等）；

##### 2）根据 标签类型 划分 不同的 类簇

爬取的数据根据名称可以分为 人物、动物、奖项、书籍、公司、事件、食物、游戏、电影、地点、角色、学校、话剧、歌曲、学术用语、物质、词语:

|    类别    | 英文名称  | 指标量  | 数量   | 筛选方式  |
| :--------: | :----:   | :----: | :----: |  :----:  |
|    人物    |  person  |  82     |  34558  |  名称、链接、简介、中文名、外文名、国籍、出生地、出生日期、职业、中文名称、类型、代表作品、民族、主要成就、别名、毕业院校、身高、星座、性别、别称、体重、逝世日期、本名、所处时代、血型、经纪公司、去世时间、民族族群、出生时间、主要奖项、出处、字号、登场作品、运动项目、属性、信仰、其他名称、职务、语言、籍贯、主要作品、英文名、所属运动队、谥号、配音、职称、官职、饰演、年龄、场上位置、位置、唱片公司、父亲、身份、生日、地位、祖籍、爵位、球衣号码、重要事件、专业特点、合作人物、妻子、政治面貌、配偶、生肖、庙号、在位时间、朝代、惯用脚、母亲、爱好、原名、特长、年号、学历、丈夫、学位/学历、nba选秀、儿子、专业方向、追赠 |
|    动物    |  animal |  25  |   1703  | 名称、链接、简介、中文名、别称、拼音、界、门、纲、科、目、属、拉丁学名、种、英文名、亚纲、分布区域、亚门、亚科、亚目、命名者及年代、族、保护级别、亚种、aid |
|    奖项    |    award    |  9   |   460  | 名称、链接、简介、中文名、外文名、类型、创办时间、时间 |
|    书籍    |   book      |  30   |   4062  | 名称、链接、简介、中文名、外文名、中文名称、类型、外文名称、作者、类别、出处、其他名称、作品名称、出版社、书名、创作年代、出版时间、isbn、地区、文学体裁、定价、页数、作品出处、原版名称、装帧、开本、又名、作品别名、字数、作品体裁 |
|  公司  |   company  | 16  |   1564  | 名称、链接、简介、中文名、外文名、类型、外文名称、简称、性质、成立时间、总部地点、公司名称、经营范围、公司类型、官网、员工数 |
|  事件  |   event  | 14  |   1251  | 名称、链接、简介、中文名、外文名、类型、别称、性质、时间、英文名、地点、结果、参战方、主要指挥官 |
|  食物  |   food  | 7  |   109  | 名称、链接、简介、中文名、分类、主要食材、口味 |
|  游戏  |   game  | 8  |   167  | 名称、链接、简介、中文名、其他名称、地区、原版名称、游戏类型 |
|  电影  |   movie  | 49  |   11107  | 名称、链接、简介、中文名、外文名、类型、代表作品、导演、制片地区、主演、编剧、上映时间、出品公司、出品时间、对白语言、片长、集数、色彩、其它译名、发行公司、制片人、拍摄地点、首播时间、在线播放平台、发行时间、每集长度、主要奖项、imdb编码、其他名称、语言、拍摄日期、分级、地区、票房、制片成本、原版名称、监制、首播平台、原作、主要配音、音乐、动画制作、出品人、角色设计、播放期间、出品、首播电视台、发行日期、网络播放 |
|  电影  |   place  | 101  |   13471  | 名称、链接、简介、中文名、外文名、中文名称、别名、地理位置、所属地区、外文名称、面积、人口、行政区类别、别称、著名景点、拼音、电话区号、气候条件、车牌代码、下辖地区、邮政区码、政府驻地、类别、火车站、方言、简称、机场、出处、运动项目、释义、性质、成立时间、属性、创办时间、时间、语言、开放时间、地区生产总值、门票价格、英文名称、英文名、地点、地址、特点、地区、行政区划代码、所属国家、主管部门、行政代码、气候类型、景点级别、现任领导、所属城市、位置、适宜游玩季节、建议游玩时长、地位、主要民族、主要城市、所属洲、官方语言、首都、政治体制、货币、国家领袖、国家、主要宗教、国土面积、人口数量、时区、人口密度、中文队名、人均gdp、位于、外文队名、角逐赛事、批准时间、主场馆、主要荣誉、现任主教练、海拔、人均生产总值、容纳人数、车站地址、隶属、级别、区域管理、国歌、发源地、国庆日、知名人物、建筑面积、道路通行、车站等级、拥有者、gdp、gdp总计、国家代码、主要线路、国际域名缩写、国际电话区号 |
|  角色  |   role  | 8  |   236  | 名称、链接、简介、中文名、外文名、登场作品、其他名称、饰演|
|  学校  |   school  | 30  |   1532  | 名称、链接、简介、中文名、外文名、类型、所属地区、类别、主要奖项、简称、属性、创办时间、英文名、地址、占地面积、校训、主管部门、现任领导、知名校友、现任校长、院系设置、学校类型、校歌、主要院系、硕士点、院校代码、学校地址、本科专业、博士点、校庆日|
|  话剧  |   show  | 56  |   5932  | 名称、链接、简介、中文名、外文名、中文名称、类型、别名、导演、地理位置、外文名称、别称、著名景点、发行公司、首播时间、类别、在线播放平台、发行时间、每集长度、歌曲原唱、简称、填词、谱曲、所属专辑、歌曲语言、音乐风格、属性、歌曲时长、语言、开放时间、编曲、门票价格、英文名称、地点、占地面积、所属国家、气候类型、景点级别、所属城市、唱片公司、适宜游玩季节、建议游玩时长、专辑歌手、监制、专辑语言、曲目数量、主持人、播出时间、制作人、播出频道、mv导演、国家/地区、制作公司、播出状态、发行地区、主办单位|
|  歌曲  |   song  | 16  |   376  | 名称、链接、简介、中文名、外文名称、发行时间、歌曲原唱、填词、谱曲、所属专辑、歌曲语言、音乐风格、歌曲时长、编曲、唱片公司、专辑歌手|
|  学术用语  |   theory  | 7  |   365  | 名称、链接、简介、中文名、外文名、学科、领域|
|  物质  |   thing  | 8  |   246  | 名称、链接、简介、中文名、外文名、组成、应用、用途 |
|  词语  |   word  | 24  |   4833  | 名称、链接、简介、中文名、外文名、类型、别名、别称、拼音、类别、注音、出处、释义、性质、属性、分类、解释、定义、含义、近义词、词性、读音、部首、反义词|

#### step 3: 将数据 导入 ES

#### step 4: 利用 python 编写 后台，并 对 ES 进行查询，返回接口数据

#### step 5: ES 数据前端展示

![ES前端展示界面](data/img/web.png)

#### step 6：实体消歧，（地点、人物等）
#### step 7：实体归一，（地点、人物等）
#### step 8：各类实体关系 实体库 和 关系库 构建

#### step 9：数据 导入 neo4j 图数据库

![知识图谱](data/img/adPv5hSL.png)

#### step 10: 编写 查询 接口，并用于 前台 展示


## 工作二 ： ChineseEDA 中文 数据增强 

### 一、项目介绍

该工作 复现了 常用 的 中文文本数据增强 策略；

### 二、功能介绍

- 同义词替换：替换一个语句中的n个单词为其同义词
- 随机插入：随机在语句中插入n个词；
- 随机交换：随机交货句子中的两个词
- 随机删除：以概率p删除语句中的词
- 基于Markov Chain的极简数据增强方法

## 工作三 ： TextClassifier 中文文本分类任务

### 一、项目介绍

本项目为基于TextCNN，TextRNN 、TextRCNN、Bi-LSTM + Attention、Adversarial LSTM、Transformer和NLP中预训练模型构建的多个常见的文本分类模型。

### 二、requirements

- python >= 3.6
- tensorflow == 1.x

### 三、文件目录

- data/：数据集
- layers/ ： 模型
  - textrnn.py
  - textcnn.py
  - BiLSTMAttention.py
  - AdversarialLSTM.py
  - Transformer.py
- model/ ： 模型保存路径
- summarys/ ：tensorboard
- tools/ ：工具目录
  - metrics.py：评估函数
  - utils.py：数据预处理
- train_word2vec_model.py：词向量预训练
- fastTextStudy.ipynb：fastText 模型
- config.py：参数配置
- TextClassifier.ipynb：主模块（项目入口）
- predict.py：预测

### 四、方法介绍

#### word2vec 词向量预训练 

##### 介绍

该方法来自于论文 【[T. Mikolov, I. Sutskever, K. Chen, G. Corrado, and J. Dean. Distributed Representations of Words and Phrases and their Compositionality. NIPS 2013](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)】、【[Le Q, Mikolov T. Distributed representations of sentences and documents[C]//International Conference on Machine Learning. 2014: 1188-1196.](https://arxiv.org/abs/1405.4053)】

##### 思路介绍

![糟糕！图片都去哪了？](data/img/微信截图_20200727223323.png)

- CBOW
  - CBOW 是用上下文预测这个词
  - 举例：以 {“The”, “cat”, “over”, “the”, “puddle”} 为上下文，能够预测或产生它们中心的词语”jumped”。模型输入为 x(c)，模型输出为 y，y 就是中心词 ‘jumped’。对于每个词语 wi 学习了两个向量。
- Skip-gram
  - Skip-gram 是预测一个词的上下文
  - 举例:以中心的词语 ”jumped” 为输入，能够预测或产生它周围的词语 ”The”, “cat”, “over”, “the”, “puddle” 等。这里我们叫 ”jumped” 为上下文。我们把它叫做Skip-Gram 模型。

- 具体介绍：【[从one-hot到word2vec](book/从one-hot到word2vec.pdf)】

#### fastText 文本分类 

##### 介绍

fastText 是Facebook 在2016年提出的一个文本分类算法，原理可参照下面论文【[LBag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)】

##### 思路介绍

![糟糕！图片都去哪了？](data/img/微信截图_20200727224807.png)

模型结构如下图所示，每个单词通过嵌入层可以得到词向量，然后将所有词向量平均可以得到文本的向量表达，在输入分类器，使用softmax计算各个类别的概率。 

需要注意以下几个问题，：

- fastText没有使用预先训练好的词嵌入层；

- 当类别很多时，fastText使用hierarchicla softmax加快计算速度；

- fastText采用n-gram额外特征来得到关于局部词顺序的部分信息,用hashing来减少N-gram的存储。

关于如何使用n-gram来获取额外特征，这里可以用一个例子来说明： 

对于句子：“我 想 喝水”, 如果不考虑顺序，那么就是每个词，“我”，“想”，“喝水”这三个单词的word embedding求平均。如果考虑2-gram, 那么除了以上三个词，还有“我想”，“想喝水”等词。对于N-gram也是类似。但为了提高效率，实际中会过滤掉低频的 N-gram。否则将会严重影响速度

- 具体介绍：
  - 【[fastText介绍](book/fastText介绍)】

#### TextCNN 文本分类

##### 介绍

该方法来自于论文 【[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)】

##### 思路介绍

![糟糕！图片都去哪了？](data/img/微信截图_20200727214604.png)

该论文提出的三种feature size 的卷积核可以认为是对应了3-gram，4-gram和5-gram 。整体模型结构如下，先用不同尺寸（3， 4， 5）的卷积核去提取特征，在进行最大池化，最后将不同尺寸的卷积核提取的特征拼接在一起作为输入到softmax中的特征向量。


#### TextRNN 文本分类

##### 介绍

该方法来自于论文 【[Recurrent Neural Network for TextClassification with Multi-Task Learning](https://arxiv.org/abs/1605.05101)】

##### 思路介绍

![糟糕！图片都去哪了？](data/img/微信截图_20200727215358.png)

结构：降维--->双向lstm ---> concat输出--->平均 -----> softmax

#### Bi-LSTM + Attention 文本分类

##### 介绍

该方法来自于论文 【[Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://aclweb.org/anthology/Y/Y15/Y15-1009.pdf)】

##### 思路介绍

![糟糕！图片都去哪了？](data/img/微信截图_20200727215616.png)

Bi-LSTM + Attention 就是在Bi-LSTM的模型上加入Attention层，在Bi-LSTM中我们会用最后一个时序的输出向量 作为特征向量，然后进行softmax分类。Attention是先计算每个时序的权重，然后将所有时序 的向量进行加权和作为特征向量，然后进行softmax分类。在实验中，加上Attention确实对结果有所提升。

#### RCNN 文本分类

##### 介绍

该方法来自于论文 【[Recurrent Convolutional Neural Networks for Text Classification](https://arxiv.org/abs/1609.04243)】

##### 思路介绍

![糟糕！图片都去哪了？](data/img/微信截图_20200727215740.png)

RCNN 整体的模型构建流程如下：

- 1）利用Bi-LSTM获得上下文的信息，类似于语言模型。

- 2）将Bi-LSTM获得的隐层输出和词向量拼接[fwOutput, wordEmbedding, bwOutput]。

- 3）将拼接后的向量非线性映射到低维。

- 4）向量中的每一个位置的值都取所有时序上的最大值，得到最终的特征向量，该过程类似于max-pool。

- 5）softmax分类。

#### Adversarial LSTM 文本分类

##### 介绍

该方法来自于论文 【[Adversarial Training Methods For Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725)】

##### 思路介绍

![糟糕！图片都去哪了？](data/img/微信截图_20200727215928.png)

上图中左边为正常的LSTM结构，右图为Adversarial LSTM结构，可以看出在输出时加上了噪声。

　　Adversarial LSTM的核心思想是通过对word Embedding上添加噪音生成对抗样本，将对抗样本以和原始样本 同样的形式喂给模型，得到一个Adversarial Loss，通过和原始样本的loss相加得到新的损失，通过优化该新 的损失来训练模型，作者认为这种方法能对word embedding加上正则化，避免过拟合。

#### Transformer  文本分类

##### 介绍

该方法来自于论文 【[Attention Is All You Need](hhttps://arxiv.org/abs/1706.03762)】

##### 思路介绍

![糟糕！图片都去哪了？](data/img/微信截图_20200727220141.png)

- Transformer 整体结构：
  - encoder-decoder 结构
- 具体介绍：
  - 左边是一个 Encoder;
  - 右边是一个 Decoder;
  
【[【关于Transformer】 那些的你不知道的事](https://gitee.com/km601/nlp_paper_study/tree/master/transformer_study/Transformer)】

## 工作四 ：text_feature_extraction 文本特征提取

### 一、介绍

- 方法类别介绍
  - TF-IDF算法关键词提取算法
  - TextRank算法关键词提取算法
  - LDA主题模型关键词提取算法
  - 互信息关键词提取算法
  - 卡方检验关键词提取算法
  - 基于树模型的关键词提取算法

### 二、 TF-IDF关键词提取算法

#### 理论基础

##### 介绍

-  类型：一种统计方法
-  作用：用以评估句子中的某一个词（字）对于整个文档的重要程度；
-  重要程度的评估：
   -  对于 句子中的某一个词（字）随着其在整个句子中的出现次数的增加，其重要性也随着增加；（正比关系）【体现词在句子中频繁性】
   -  对于 句子中的某一个词（字）随着其在整个文档中的出现频率的增加，其重要性也随着减少；（反比关系）【体现词在文档中的唯一性】
- 重要思想：
  - 如果某个单词在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类；

##### 计算公式

- 词频 （Term Frequency，TF）
  - 介绍：体现 词 在 句子 中出现的频率；
  - 问题：
    - 当一个句子长度的增加，句子中 每一个 出现的次数 也会随之增加，导致该值容易偏向长句子；
    - 解决方法：
      - 需要做归一化（词频除以句子总字数）
  - 公式

![](data/img/20200809105640.png)

- 逆文本频率(Inverse Document Frequency，IDF)
  - 介绍：体现 词 在文档 中出现的频率
  - 方式：某一特定词语的IDF，可以由总句子数目除以包含该词语的句子的数目，再将得到的商取对数得到；
  - 作用：如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力
  - 公式：

![](data/img/20200809110034.png)

- TF-IDF
    - 介绍：某一特定句子内的高词语频率，以及该词语在整个文档集合中的低文档频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语。
    - 优点：
      - 容易理解；
      - 容易实现；
    - 缺点：
      - 其简单结构并没有考虑词语的语义信息，无法处理一词多义与一义多词的情况

![](data/img/20200809110358.png)

- 应用
  - 搜索引擎；
  - 关键词提取；
  - 文本相似性；
  - 文本摘要

### 二、PageRank算法

#### 理论学习

- 论文：[The PageRank Citation Ranking: Bringing Order to the Web](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf)
- 介绍：通过计算网页链接的数量和质量来粗略估计网页的重要性，算法创立之初即应用在谷歌的搜索引擎中，对网页进行排名；
- 核心思想：
  - 链接数量：如果一个网页被越多的其他网页链接，说明这个网页越重要，即该网页的PR值（PageRank值）会相对较高；
  - 链接质量：如果一个网页被一个越高权值的网页链接，也能表明这个网页越重要，即一个PR值很高的网页链接到一个其他网页，那么被链接到的网页的PR值会相应地因此而提高；
- 计算公式

![](data/img/20200809121756.png)

> $S(V_i)$ ： 网页 i 的 重要性；
> 
> d：托尼系数；
> 
> $ln(V_i)$：整个互联网中所存在的有指向网页 i 的链接的网页集合；
> 
> $Out(V_j)$： 网页 j 中存在的指向所有外部网页的链接的集合；
> 
> $Out(V_j)$：该集合中元素的个数；

### 三、TextRank算法

- 论文：[TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
- 介绍：一种基于图的用于关键词抽取和文档摘要的排序算法，由谷歌的网页重要性排序算法PageRank算法改进而来，它利用一篇文档内部的词语间的共现信息(语义)便可以抽取关键词，它能够从一个给定的文本中抽取出该文本的关键词、关键词组，并使用抽取式的自动文摘方法抽取出该文本的关键句；
- 基本思想：将文档看作一个词的网络，该网络中的链接表示词与词之间的语义关系；
- 计算公式：

![](data/img/20200809122329.png)

- pageRank vs TextRank
  - PageRank算法根据网页之间的链接关系构造网络，TextRank算法根据词之间的共现关系构造网络；
  - PageRank算法构造的网络中的边是有向无权边，TextRank算法构造的网络中的边是无向有权边。

## 工作五 NERer 中文命名实体识别

### 一、项目介绍

该项目 中文命名实体识别 任务

### 二、目录介绍

- LSTM_IDCNN：LSTM-CRF 中文命名实体识别模型
- LSTM_IDCNN：IDCNN-CRF 中文命名实体识别模型
- bert_crf：bert_crf 中文命名实体识别模型
- albert_crf：albert_crf 中文命名实体识别模型

## 工作六 TextMatching 中文 文本匹配 方法

### 一、项目介绍

本项目包含目前大部分文本匹配模型，数据集为 data，训练数据10w条，验证集和测试集均为1w条

### 二、目录介绍

- graph：模型
  - dssm.py
  - abcnn.py
  - bimpm.py
  - convnet.py
  - diin.py
  - drcn.py
  - esim.py
- data：数据
- output：结果
- utils：
  - load_data.py
  - data_utils.py
- config.json 参数 配置
- main.py
- main_static.py
- main_dynamic.py
- main_dynamic_static.py
- test.py
- test_static.py
- test_dynamic.py
- word2vec_static.py：该版本是采用gensim来训练词向量
- word2vec_dynamic.py：该版本是采用tensorflow来训练词向量，训练完成后会保存embedding矩阵、词典和词向量在二维矩阵的相对位置的图片

### 三、模型效果对比

模型 | loss | acc | 输入说明 | 论文地址
:-: | :-: | :-: | :-: | :-: |
DSSM | 0.7613157 | 0.6864 | 字向量 | [DSSM](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf) |
ConvNet | 0.6872447 | 0.6977 | 字向量 | [ConvNet](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.6492&rep=rep1&type=pdf) |
ESIM | 0.55444807| 0.736 | 字向量 | [ESIM](https://arxiv.org/pdf/1609.06038.pdf) |
ABCNN | 0.5771452| 0.7503 | 字向量 | [ABCNN](https://arxiv.org/pdf/1512.05193.pdf) |
BiMPM | 0.4852| 0.764 | 字向量+静态词向量 | [BiMPM](https://arxiv.org/pdf/1702.03814.pdf) |
DIIN | 0.48298636| 0.7694 | 字向量+动态词向量 | [DIIN](https://arxiv.org/pdf/1709.04348.pdf) |
DRCN | 0.6549849 | 0.7811 | 字向量+静态词向量+动态词向量+是否有相同词 | [DRCN](https://arxiv.org/pdf/1805.11360.pdf) |

## 工作七 textSummarization 中文 文本摘要 方法

### 一、介绍

本方法的摘要生成是抽取式，通过把段落训练词向量，得到句子的向量。再通过pagerank方法得到权重高的向量，从而得到对应的句子

### 二、目录介绍

- [textrank](textrank/)：基于 textrank 的 文本摘要 方法

### 三、 安装环境

- math<br>
- numpy<br>
- jieba<br>
- gensim<br>
- networkx<br>
- itertools<br>
- textrank4zh<br>

### 四、 示例

word2vec_textranl/word2vec_textrank.py:<br>

    content = '原标题：专访：俄方希望与中方寻找双边贸易新增长点——访俄罗斯工业和贸易部长曼图罗夫新华社记者栾海高兰<br>
    “在当前贸易保护主义抬头背景下，俄方希望与中方共同应对风险，化消极因素为机遇，寻找俄中贸易的新增长点”，<br>
    俄罗斯工业和贸易部长丹尼斯·曼图罗夫日前在接受     新华社记者专访时说。曼图罗夫表示，中国一直是俄重要的战略协作伙伴。<br>
    当前俄中关系保持快速发展，双方不断在贸易和工业领域寻找新的合作点。据他介绍，今年1月至7月，俄中双边贸易额同比增长超25%，达近600亿美元。<br>
    曼图罗夫说，俄中两国正在飞机轮船和其他交通工具制造、无线电设备研发、制药和化工等工业领域开展合作。俄中投资基金支持了两国众多开发项目，投资方对该基金继续注资的兴趣十分浓厚。<br>
    在回顾日前结束的第四届东方经济论坛时，曼图罗夫表示，这一论坛已成为俄与中国和其他东北亚国家讨论重大经济合作议题的平台。<br>
    “在本届论坛期间，俄方与海外企业共签署220项各类协议，协议总金额达3.1万亿卢布（1美元约合66卢布）”。<br>
    曼图罗夫说，俄工业和贸易部在本届论坛上与俄外贝加尔边疆区的一家矿业公司负责人进行磋商，以落实中方企业持有该公司股份的相关事宜。根据相关协议，俄中企业将在外贝加尔边疆区的金矿区联合勘探。<br>
    据俄方估算，这一俄中合作项目有望年产黄金约6.5吨，在2020年前使该边疆区贵金属开采量比目前增加约40%，从而有力促进当地经济发展。责任编辑：张义凌'
print(do(content))<br>

运行结果如下：

     关键句：
     当前俄中关系保持快速发展，双方不断在贸易和工业领域寻找新的合作点。<br>
     据他介绍，今年1月至7月，俄中双边贸易额同比增长超25%，达近600亿美元。俄中投资基金支持了两国众多开发项目，投资方对该基金继续注资的兴趣十分浓厚。


​     
 textrank4zh.py:<br>
 运行结果如下：
        摘要：
        0 0.10636689669924555 原标题：专访：俄方希望与中方寻找双边贸易新增长点——访俄罗斯工业和贸易部长曼图罗夫新华社记者栾海高兰 “在当前贸易保护主义抬头背景下，俄方希望与中方共同应对风险，化消极因素为机遇，寻找俄中贸易的新增长点”，俄罗斯工业和贸易部长丹尼斯·曼图罗夫日前在接受新华社记者专访时说<br>
        8 0.0961579730882088 曼图罗夫说，俄工业和贸易部在本届论坛上与俄外贝加尔边疆区的一家矿业公司负责人进行磋商，以落实中方企业持有该公司股份的相关事宜<br>
        4 0.09384810578387712 曼图罗夫说，俄中两国正在飞机轮船和其他交通工具制造、无线电设备研发、制药和化工等工业领域开展合作<br>

## 工作八 QAer 中文 问答

### 一、 项目介绍

FAQ 问答系统构建

### 二、目录介绍

- textMatch
  - graph：模型
    - dssm.py
    - abcnn.py
    - bimpm.py
    - convnet.py
    - diin.py
    - drcn.py
    - esim.py
  - utils：
    - load_data.py
    - data_utils.py
- data
- output
- tools
  - bert_tools.py              bert-as-service 基本操作
  - common_tools.py            通用操作，eg： 装饰器 之 计数器
  - io_tools.py                IO 操作
  - KeywordProcessor.py        FlashText 操作
  - loader.py                  数据 加载 存储
  - multiprocessing_tools.py   并行化 工具
  - sim_tools.py               相似度 计算工具
  - text_preprocess.py         文本预处理
  - v1
    - S1利用fastText做文本分类.ipynb
    - S2数据预处理导入ES.ipynb
    - S3TFidf计算句向量.ipynb
    - s4利用Bert做词向量相似度.ipynb
    - s4利用DL计算相似度.ipynb
    - QA系统构建.ipynb
  - v2
    - s1_cutword_ngram.py
    - s1_ftCleanStopWord.py
    - S2_fastText做文本分类.ipynb
    - S3数据预处理导入ES.ipynb
    - S4_BM25.ipynb
    - S4TFidf计算句向量.ipynb
    - s5利用Bert做词向量相似度.ipynb
    - s6_利用DL计算相似度.ipynb
    - QA系统构建.ipynb

### 三、项目思路介绍

![](data/img/20200920223420.png)

![](img/data/20200920223504.png)

## 工作九 seq2seqAttn 解决不同 的 生成式 任务

### 一、介绍

利用 seq2seqAttn 模型 解决不同 的 生成式 任务，包括 问答系统、文本摘要、文本翻译；

### 二、目录介绍

- seq2seqAttn
  - Config.py
  - GreedySearchDecoder.py
  - loader.py
  - main.py
  - model.py
  - test.py
  - utils.py
  - Voc.py
  - main.ipynb
  - test.ipynb
  - data
    - TextSummarization 文本摘要数据
    - qa 数据
  - seq2seq_model_origin.ipynb 原始版本
  - seq2seq_model.ipynb
  - tools.py

### 三、项目进度

1. QA：可以使用，但是结果有问题；
2. summary: 可以使用，结果有问题；
3. 文本翻译：（未完成）

## 工作十 rasa 学习

### 介绍

rasa 学习

### 目录介绍

1. [【关于 rasa 】那些你不知道的事](https://github.com/km1994/nlp_paper_study/tree/master/DialogueSystem_study/rasa)
   1. [【关于 rasa 安装 】那些你不知道的事](https://github.com/km1994/nlp_paper_study/tree/master/DialogueSystem_study/rasa/rasa安装手册.md)
   2.  [【关于 rasa 基本架构 】那些你不知道的事](https://github.com/km1994/nlp_paper_study/tree/master/DialogueSystem_study/rasa/基本框架.md)
   3.  [【关于 rasa中文对话系统】那些你不知道的事](https://github.com/km1994/nlp_paper_study/tree/master/DialogueSystem_study/rasa/rasa中文对话系统.md)
