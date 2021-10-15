# 【关于 IRN 】 那些的你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> NLP 面经地址：https://github.com/km1994/NLP-Interview-Notes
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 
> 论文：ScriptWriter: Narrative-Guided Script Generation
> 
> 发表会议：ACL2020
> 
> 论文地址：https://www.aclweb.org/anthology/2020.acl-main.10/
> 
> github：#

## 一、论文摘要

- 论文动机：如何将输入中对话状态的slot-value对正确的在response生成
- 论文方法：
  - 迭代网络：来不断修正生成过程不对的slot-value；
  - 强化学习：不断更新，实验证明我们的网络生成的回复中中slot关键信息生成的正确性大大提高。
- 实验结果：对多个基准数据集进行了综合研究，结果表明所提出的方法显著降低了所有强基线的时隙错误率。人类的评估也证实了它的有效性。

