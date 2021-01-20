# 【关于 KBQA】那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 目录

![](img/微信截图_20210121004759.png)


## 一、基于词典和规则的方法

### 1.1 代表项目

1. [豆瓣影评问答](https://github.com/weizhixiaoyi/DouBan-KGQA) 
2. [基于医疗知识图谱的问答系统](https://github.com/zhihao-chen/QASystemOnMedicalGraph)

### 1.2 流程

#### 1.2.1. 句子输入

```s
  - eg：query：高血压要怎么治？需要多少天？
```

#### 1.2.2. 问句解析

- 实体抽取：
  - 作用：得到匹配的词和类型
  - 方法：
    - 模式匹配：
      - 介绍：主要采用规则 提槽
      - 工具：正则表达式
    - 词典：
      - 介绍：利用词典进行匹配
      - 采用的词典匹配方法：trie和Aho-Corasick自动机，简称AC自动机 
      - 工具：ahocorasick、FlashText 等 python包
    - 基于词向量的文本相似度计算：
      - 介绍：计算 query 中 实体 与 实体库 中候选实体的相似度，通过设定阈值，得到最相似的 实体
      - 工具：词向量工具（TF-idf、word2vec、Bert 等）、相似度计算方法（余弦相似度、L1、L2等）
    - 命名实体识别方法：
      - 利用 命名实体识别方法 识别 query 中实体
      - 方法：BiLSTM-CRF等命名实体识别模型 
    - 举例说明：

```s
  eg：通过解析 上面的 query ，获取里面的实体和实体类型：{'Disease': ['高血压'], 'Symptom': ['高血压'], 'Complication': ['高血压']}
```

- 属性和关系抽取：
  - 作用：抽取 query 中 的 属性和关系
  - 方法：
    - 模式匹配：
      - 介绍：主要采用规则匹配
      - 工具：正则表达式
    - 词典：
      - 介绍：利用词典进行匹配
      - 采用的词典匹配方法：trie和Aho-Corasick自动机，简称AC自动机
      - 工具：ahocorasick、FlashText 等 python包
    - 意图识别方法：
      - 介绍：采用分类模型 对 query 所含关系 做预测
      - 工具：
        - 机器学习方法：LR、SVM、NB
        - 深度学习方法：TextCNN、TextRNN、Bert 等
    - 命名实体识别方法：【同样，可以采用命名实体识别挖掘出 query 中的某些动词和所属类型】
      - 利用 命名实体识别方法 识别 query 中实体
      - 方法：BiLSTM-CRF等命名实体识别模型 
  - 举例说明：

```s
- eg：通过解析 上面的 query ，获取里面的实体和实体类型：
     - predicted intentions:['query_period']  高血压要怎么治？
     - word intentions:['query_cureway']      需要多少天？
```

#### 1.2.3. 查询语句生成

- 作用：根据 【问句解析】 的结果，将 实体、属性和关系转化为对于的 图数据库（eg：Neo4j图数据库等）查询语句
- 举例说明：

```s
- eg： 
  对于 query：高血压要怎么治？需要多少天？
- sql 解析结果：
  [{'intention': 'query_period', 'sql': ["MATCH (d:Disease) WHERE d.name='高血压' return d.name,d.period"]}, {'intention': 'query_cureway', 'sql': ["MATCH (d:Disease)-[:HAS_DRUG]->(n) WHERE d.name='高血压' return d.name,d.treatment,n.name"]}]
```

#### 1.2.4. 查询数据库和结果生成

- 作用：利用 【查询语句生成】 的结果，去 图数据库 中 查询 答案，并利用预设模板 生成答案
  
```s
- eg:
   - 高血压可以尝试如下治疗：药物治疗;手术治疗;支持性治疗
```

## 二、基于信息抽取的方法

### 2.1 代表项目

- [Knowledge Based Question Answering](https://github.com/wudapeng268/KBQA-Baseline)
- CCKS：[ccks2019-ckbqa-4th-codes](https://github.com/duterscmy/ccks2019-ckbqa-4th-codes)、[CCKS2018 CKBQA 1st 方案](https://github.com/songlei1994/ccks2018)、[中文知识图谱问答 CCKS2019 CKBQA - 参赛总结](https://blog.nowcoder.net/n/630128e8e6dd4be5947adbfde8dcea44)
- NLPCC：[NLPCC2016 KBQA 1st 方案](https://github.com/huangxiangzhou/NLPCC2016KBQA)

### 2.2 思路

#### 2.2.1. 分类单跳和多跳问句

- 思路：利用 文本分类方法 对 query 进行分类，判断其属于 一跳问题还是多跳问题
- 方法：文本分类方法【TextCNN、TextRNN、Bert 等】
- 解析

> 单跳：SPARQL 只出现一个三元组

```s
    q26:豌豆公主这个形象出自于哪？
    select ?x where { <豌豆公主_（安徒生童话）> <作品出处> ?x. }
    <安徒生童话>
```

> 双跳或多跳：SPARQL 只出现两个以上三元组

```s
    q524:博尔赫斯的国家首都在哪里？
    select ?x where { <豪尔赫·路易斯·博尔赫斯_（阿根廷作家）> <出生地> ?y. ?y <首都> ?x}
    <布宜诺斯艾利斯_（阿根廷的首都和最大城市）>
```

#### 2.2.2. 分类链式问句（二分类）

- 思路：利用 文本分类方法 对 query 进行分类，判断其是否 属于 链式问句
- 介绍：链式：SPARQL 多个三元组呈递进关系，x->y->z，非交集关系

```s
    q894:纳兰性德的父亲担任过什么官职？
    select ?y where { <纳兰性德> <父亲> ?x. ?x <主要职位> ?y. }
    "武英殿大学士"    "太子太傅"
    q554:宗馥莉任董事长的公司的公司口号是？
    select ?y where { ?x <董事长> <宗馥莉>. ?x <公司口号> ?y. }
    "win happy health,娃哈哈就在你身边"
```

#### 2.2.3. 主谓宾分类（三分类）

- 思路：利用 文本分类方法 对 query 进行分类，判断 问句的答案对应三元组里面的 主谓宾
- 问句的答案对应三元组里面的主语，spo=0

```s
    q70:《悼李夫人赋》是谁的作品？
    select ?x where { ?x <代表作品> <悼李夫人赋>. }
    <汉武帝_（汉朝皇帝）>
```

- 问句的答案对应三元组里面的谓语，spo=1

```s
    q506:林徽因和梁思成是什么关系？
    select ?x where { <林徽因_（中国建筑师、诗人、作家）> ?x <梁思成>. }
    <丈夫>
```

- 问句的答案对应三元组里面的宾语，spo=2

```s
    q458:天津大学的现任校长是谁？
    select ?x where { <天津大学> <现任校长> ?x . }
    <李家俊_（天津市委委员，天津大学校长）>
```

#### 2.2.4. 实体提及（mention）识别 

- 思路：对于 给定 query，我们需要 识别出 query 中 所含有的 实体提及（mention）
- 问题：主办方提供的数据中 包含 ：query、sql语句，但是 sql语句中的实体并不能在 query 中被找到，如下：

```s
    q1440:济南是哪个省的省会城市？
    select ?x where { ?x <政府驻地> <济南_（山东省省会）>. }
    <山东_（中国山东省）>
```

> 注：query 中 的 济南 是 <济南_（山东省省会）>的 简称、省会城市 在 知识库中 对应 <政府驻地> 

- 解决方法：根据训练语料的SPARQL语句，查找实体的提及，反向构建训练数据
- 思路：

1. 根据 query 和 sql，查询 实体提及
2. 反向构建训练数据

```s
    q6:叔本华信仰什么宗教？
    select ?y where { <亚瑟·叔本华> <信仰> ?y. }
    <佛教>

    >>>
    叔本华信仰什么宗教？	['叔本华']
    亚瑟·叔本华信仰什么宗教？	['亚瑟·叔本华']

    >>>
    叔 本 华 信 仰 什 么 宗 教 ？	B-SEG I-SEG E-SEG O O O O O O O
    亚 瑟 · 叔 本 华 信 仰 什 么 宗 教 ？	B-SEG I-SEG I-SEG I-SEG I-SEG E-SEG O O O O O O O
```


## 参考资料

1. [中文知识图谱问答 CCKS2019 CKBQA - 参赛总结](https://blog.nowcoder.net/n/630128e8e6dd4be5947adbfde8dcea44)