# 【关于 Prompt Tuning】 那些你不知道的事

> 作者：杨夕
> 
> NLP论文学习笔记：https://github.com/km1994/nlp_paper_study
> 
> **[手机版NLP论文学习笔记](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100005719&idx=1&sn=14d34d70a7e7cbf9700f804cca5be2d0&chksm=1bbff26d2cc87b7b9d2ed12c8d280cd737e270cd82c8850f7ca2ee44ec8883873ff5e9904e7e&scene=18#wechat_redirect)**
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 
> NLP 百面百搭 地址：https://github.com/km1994/NLP-Interview-Notes
> 
> **[手机版NLP百面百搭](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100005719&idx=3&sn=5d8e62993e5ecd4582703684c0d12e44&chksm=1bbff26d2cc87b7bf2504a8a4cafc60919d722b6e9acbcee81a626924d80f53a49301df9bd97&scene=18#wechat_redirect)**
> 
> 推荐系统 百面百搭 地址：https://github.com/km1994/RES-Interview-Notes
> 
> **[手机版推荐系统百面百搭](https://mp.weixin.qq.com/s/b_KBT6rUw09cLGRHV_EUtw)**

## 一、Prompt Tuning 发展路线

### 1.1 技术发展路线

- 特征工程阶段
  - 依赖大量人工
  - 需要监督数据
- 架构工程阶段
  - 人工构建特征的工作量减少
  - 设计各异的神经网络结构从而拿到结果需要监督数据
- 预训练-微调阶段
  - 预训练可以不需要监督数据

### 1.2 工业界 vs 学术界

- 工业界:更适合工业界的少数据场景，减少一部分标注成本
  - 工业界对于少数据场景，往往(也有用半监督学习) 先rule-based走通业务逻辑，在线上线下标注数据，再进一步监督学习或微调。
  - 目前工业界里还难打的过微调
  - 定位:可以探索
- 学术界:适合学术研究
  - 新任务+PL，效果不要太差，都是一次新的尝试与探索
  - 定位:做研究

## 二、Prompt Tuning 工作流程介绍

### 2.1 文本代入 Template

- 介绍：模板Template 构造，文本代入 Template
- 思路：
  - step 1: 构建模板：[x] overall,it was a[z] movie.
  - step 2: 构建文本：l love this movie.
  - step 3: 文本代入 Template: l love this movie. overall, it was a [z] movie

### 2.2 映射 Verbalizer 构造

- 介绍：建立预测词-标签的映射关系
- eg:

```s
    fantastic、great、amazing -> positive
    boring、bad -> negative
```

### 2.3 Prediction预测

- 介绍：根据 Verbalizer ，使用预训练模型对 2.1 代后的文本进行预测
- eg:
  - I love this movie.overallit was a **fantastic** movie

### 2.4 Mapping映射

- 介绍：预测结果代入Verbalizer，得到映射后的结果
- eg：

```s
    fantastic -> positive
```

## 三、Prompt Tuning 研究方向介绍

### 3.1 Template 设计研究

> eg: I love this movie. overall, it was a [z] movie

#### 3.1.1 Template 形状研究

1. cloze prompt
   1. 介绍：[z] 在句中，适合使用了Mask任务的LM
2. prefix prompt
   1. 介绍：[z] 在句末，适合生成LM、自回归LM （自编码LM（Bert） vs 自回归LM（GPT））
3. 文本匹配任务，Prompt可以有两个[X]

#### 3.1.2 Template 设计研究

1. 手工设计

- 介绍：人工 手工设计 Template
- 优点：直观
- 缺点：成本高，需要实验、经验等

2. 自动学习 模Template板

- 介绍：通过模型学习上下文，自动生成 Template
- 离散Prompt
  - 介绍：自动生成自然语言词
  - eg: 给定一个大的文本库，给定输入x和输出y，在文本库中离散地搜索出现频繁的中间词或连词等，从而得到一个模板。
- 连续Prompt
  - 介绍：Template的设计不必拘泥于自然语言，直接变成embedding表示也是可以的，设置独立于LM的模板参数，可以根据下游任务进行微调
  - eg：给定一个可训练的参数矩阵，将该参数矩阵与输入文本进行连接，从而丢入模型进行训练。

### 3.2 Verbalizer 设计研究 (Answer Engineering)

- 介绍：寻找合适的答案空间Z，以及答案与标签的映射
- eg：Knowledgeable Prompt-tuning:Incorporating Knowledge intoPrompt Verbalizer for Text Classification (KPT)
  - 用KB去查询Label相关词作为候选集，然后去噪

![](img/20230223073844.png)

### 3.3 训练策略(Prompt-based Training Strategies)

- Tuning-free Prompting
  - 直接做zero-shot
- Fixed-LM Prompt Tuning
  - 引入额外与Prompt相关的参数，固定LM参数，微调与Prompt相关参数
- Fixed-prompt LM Tuning
  - 引入额外与Prompt相关的参数，固定与Prompt相关参数，微调LM
- Prompt + LM Tuning
  - 引入额外与Prompt相关的参数，两者都微调

## 四、prompt进阶——自动学习prompt

### 4.1 动机

手工设计prompt（基于token的prompt）还有一个问题是，**模型对prompt很敏感，不同的模板得到的效果差别很大**。

![](img/20230223074901.png)
> 注：prompt一字之差效果也会差别很大 (来自文献[2])

所以研究学者就提出自动学习prompt向量的方法。因为我们输入进去的是人类能看懂的自然语言，那在机器眼里是啥，啥也不是， 也不能这么说吧，prompt经过网络后还是得到一个个向量嘛，既然是向量，当然可以用模型来学习了，甚至你输入一些特殊符号都行，模型似乎无所不能，什么都能学，只要你敢想，至于学得怎么样，学到了什么，还需进一步探究。

### 4.2 P-tuning——token+vector组合成prompt [论文](https://arxiv.org/abs/2103.10385) [github](https://github.com/THUDM/P-tuning)

- 动机：手工设计prompt（基于token的prompt）存在问题，那么是否可以引入（基于vector的prompt），所以就有了 基于 token+vector 的 prompt
- 具体说明（如下图）：
  - 任务是让模型来预测一个国家的首都
  - 左边是全token的prompt，文献里称为“离散的prompt”，有的同学一听"离散"就感觉懵了，其实就是一个一个token组成的prompt就叫“离散的prompt”。
  - 右边是token+vector形式的prompt，其实是保留了原token prompt里面的关键信息(capital, Britain)，(capital, Britain)是和任务、输出结果最相关的信息，其他不关键的词汇(the, of ,is)留给模型来学习。

- token形式的prompt: “The captital of Britain is [MASK]”
- token+vector: “h_0 , h_1, ... h_i, captital, Britain, h_(i+1), ..., h_m [MASK]”

![](img/20230223075648.png)










## 参考资料

- [Prompt Learning全面梳理扫盲](https://zhuanlan.zhihu.com/p/493900047)
- [一文轻松入门Prompt(附代码)](https://zhuanlan.zhihu.com/p/440169921)
- [prompt工程指南](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [[细读经典]P-tuning：用“连续提示微调”来增强“超大规模语言模型”的下游能力](https://zhuanlan.zhihu.com/p/391992466)
- [NLPer福利！清华推出Prompt-tuning开源工具包，取代传统的微调fine-tuning](https://zhuanlan.zhihu.com/p/415944918)
- [一文跟进Prompt进展！综述+15篇最新论文逐一梳理](https://blog.csdn.net/qq_27590277/article/details/121173627)
- [Prompt在低资源NER中的应用](https://zhuanlan.zhihu.com/p/428225612)
