# 【关于 TPLinker】 那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 论文名称：《TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking》
> 
> 论文地址：https://arxiv.org/pdf/2010.13415
> 
> 源码地址：https://github.com/131250208/TPlinker-joint-extraction
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 目录


## 一、动机篇

### 1.1 什么是 实体关系联合抽取？

给定一段文本，如何抽取出里面的实体对和对应关系。

![](img/微信截图_20210123110915.png)
> 注：对于上面句子，需要抽取出实体对(ent1，rel,ent2)，例如： (United States, Country-Presideng, Trump)

这个时候需要怎么抽呢？

## 二、整体框架介绍篇

### 2.1 实体关系联合抽取 有哪些方法？

实体关系联合抽取主要分 pipeline 方法和 end2end 方法。

### 2.1.1 实体关系联合抽取 pipeline 方法是什么？存在什么问题？

* 思路：先命名实体识别（ NER） , 在 关系抽取（RE）
* 问题：
    * 忽略两任务间的相关性
    * 误差传递。NER 的误差会影响 RE 的性能

### 2.1.2 实体关系联合抽取 end2end 方法 是什么？存在什么问题？

* 解决问题：实体识别、关系分类
* 思路：
    * 实体识别
        * BIOES 方法：提升召回？和文中出现的关系相关的实体召回
        * 嵌套实体识别方法：解决实体之间有嵌套关系问题
        * 头尾指针方法：和关系分类强相关？和关系相关的实体召回
        * copyre方法
    * 关系分类：
        * 思路：判断 【实体识别】步骤所抽取出的实体对在句子中的关系
        * 方法：
            * 方法1：1. 先预测头实体，2. 再预测关系、尾实体
            * 方法2：1. 根据预测的头、尾实体预测关系
            * 方法3：1. 先找关系，再找实体 copyre
        * 需要解决的问题：
            * 关系重叠 (如例子一)
            * 关系间的交互 (如例子一)
  
> 例子一 ：(BarackObama, Governance, UnitedStates) 与 (BarackObama, PresidentOf, UnitedStates) <br/>
> 从示例可以看出，实体对(BarackObama，UnitedStates) 存在Governance 和PresidentOfPresidentOf两种关系，也就是关系重叠问题。

> 例子二 ：(BarackObama, LiveIn, WhiteHouse) 和 (WhiteHouse, PresidentialPalace, UnitedStates) -> (BarackObama, PresidentOf, UnitedStates) <br/>
> 从示例可以看出，实体对(BarackObama，WhiteHouse)  和 实体对(WhiteHouse，UnitedStates)  存在中间实体WhiteHouse，而且通过(BarackObama, LiveIn, WhiteHouse) 和 (WhiteHouse, PresidentialPalace, UnitedStates) 能够推出 (BarackObama, PresidentOf, UnitedStates) 关系，也就是关系间存在交互问题。

## 三、




## 参考资料

1. [关系抽取之TPLinker解读加源码分析](https://zhuanlan.zhihu.com/p/342300800)
2. [实体和关系联合抽取方法](https://zhuanlan.zhihu.com/p/136553137)
3. [实体关系抽取新范式！TPLinker：单阶段联合抽取，并解决暴漏偏差](https://mp.weixin.qq.com/s?__biz=MzI5NjA4MDIyMw==&mid=2656168560&idx=1&sn=324315ac515319773b570797b64f8126&chksm=f7ec8c40c09b05563d36eb4b7e064beb89edd0836cef965ad2b336e5a8712f22ba3ee9d51785&mpshare=1&scene=22&srcid=0122rjlgsq1b5FG3Inf9gfUh&sharer_sharetime=1611290064963&sharer_shareid=da84f0d2d31380d783922b9e26cacfe2#rd)





