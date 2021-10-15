# ACL 2020对话系统相关论文整理

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 

## 目录

- [【关于 对话系统 】那些你不知道的事](#关于-对话系统-那些你不知道的事)
  - [目录](#目录)
  - [对话系统有哪几种？](#对话系统有哪几种)
  - [这几类对话系统的区别是什么？](#这几类对话系统的区别是什么)
  - [面向任务的对话系统目标是什么？](#面向任务的对话系统目标是什么)
  - [对话系统的关键模块是什么？](#对话系统的关键模块是什么)
  - [为什么要有多轮对话？](#为什么要有多轮对话)
  - [对话系统如何获取 必要信息？](#对话系统如何获取-必要信息)
  - [面向任务的对话系统任务的形式化符号定义](#面向任务的对话系统任务的形式化符号定义)
  - [按照技术实现划分，对话系统可分为几类？](#按照技术实现划分对话系统可分为几类)
  - [模块化的对话系统 由哪些 模块，分别做什么？](#模块化的对话系统-由哪些-模块分别做什么)
  - [模块化的对话系统 NLU 模块介绍？](#模块化的对话系统-nlu-模块介绍)
    - [意图识别定义是什么?](#意图识别定义是什么)
    - [槽位值定义是什么?](#槽位值定义是什么)
    - [能否举个例子说明一下呢？](#能否举个例子说明一下呢)
    - [怎么将两种方式合并？](#怎么将两种方式合并)
    - [评价指标是什么？](#评价指标是什么)
  - [DST 是什么？](#dst-是什么)
    - [DST 的 输入输出是什么？](#dst-的-输入输出是什么)
    - [对话状态的表现（DST-State Representation）组成是什么？](#对话状态的表现dst-state-representation组成是什么)
    - [DST 存在问题及解决方法？](#dst-存在问题及解决方法)
    - [DST 实现方法是什么？](#dst-实现方法是什么)
  - [DPL （对话策略优化）模块是什么?](#dpl-对话策略优化模块是什么)
    - [DPL 的输入输出是什么？](#dpl-的输入输出是什么)
    - [基于规则的 DPL 方法介绍？](#基于规则的-dpl-方法介绍)
  - [NLG 模块是什么？](#nlg-模块是什么)
    - [DPL 的输入输出是什么？](#dpl-的输入输出是什么-1)
  - [文章目录](#文章目录)
  - [优质文章学习](#优质文章学习)
  - [优质代码学习](#优质代码学习)


## 摘要

【摘要】 ACL 2020 总共收录56篇Dialogue相关的论文，达到历史之最。其中对话生成相关的论文数量最多达到15篇，其次是对话新任务和数据的发布大概有7篇，对话系统评价分析相关7篇，对话状态跟踪7篇，端到端对话系统6篇，对话理解相关5篇，对话策略学习4篇，多模态相关4篇，其他1篇。

## 对话理解
1. Recursive Template-based Frame Generation for Task Oriented Dialog
2. Span-ConveRT: Few-shot Span Extraction for Dialog with Pretrained Conversational Representations
3. Coach: A Coarse-to-Fine Approach for Cross-domain Slot Filling
4. Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network
5.Unknown Intent Detection Using Gaussian Mixture Model with an Application to Zero-shot Intent Classification
## 对话状态跟踪
1. A Contextual Hierarchical Attention Network with Adaptive Objective for Dialogue State Tracking
2. Efficient Dialogue State Tracking by Selectively Overwriting Memory
3. Rethinking Dialogue State Tracking with Reasoning
4. SAS: Dialogue State Tracking via Slot Attention and Slot Information Sharing
5. Zero-Shot Transfer Learning with Synthesized Data for Multi-Domain Dialogue State Tracking
6. Dialogue State Tracking with Explicit Slot Connection Modeling
7. Modeling Long Context for Task-Oriented Dialogue State Generation
## 对话策略
1. Learning Dialog Policies from Weak Demonstrations
2. Learning Efficient Dialogue Policy from Demonstrations through Shaping
3. Multi-Agent Task-Oriented Dialog Policy Learning with Role-Aware Reward Decomposition
4. Semi-Supervised Dialogue Policy Learning via Stochastic Reward Estimation
## 对话生成
1. Data Manipulation: Towards Effective Instance Learning for Neural Dialogue Generation via Learning to Augment and Reweight
2. Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness
3. Diversifying Dialogue Generation with Non-Conversational Text
4. Generate, Delete and Rewrite: A Three-Stage Framework for Improving Persona Consistency of Dialogue Generation
5. Learning to Customize Model Structures for Few-shot Dialogue Generation Tasks
6. Multi-Domain Dialogue Acts and Response Co-Generation
7. Negative Training for Neural Dialogue Response Generation
8. Neural Generation of Dialogue Response Timings
9. Paraphrase Augmented Task-Oriented Dialog Generation
10. PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable
11. Slot-consistent NLG for Task-oriented Dialogue Systems with Iterative Rectification Network
12. Towards Holistic and Automatic Evaluation of Open-Domain Dialogue Generation
13. USR: An Unsupervised and Reference Free Evaluation Metric for Dialog Generation
14. You Impress Me: Dialogue Generation via Mutual Persona Perception
15. “None of the Above”: Measure Uncertainty in Dialog Response Retrieval
## 端到端对话系统
1. Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog
2. End-to-End Neural Pipeline for Goal-Oriented Dialogue Systems using GPT-2
3. Grounding Conversations with Improvised Dialogues
4. Large Scale Multi-Actor Generative Dialog Modeling
5. Conversational Word Embedding for Retrieval-Based Dialog System
6. Learning Low-Resource End-To-End Goal-Oriented Dialog for Fast and Reliable System Deployment
## 对话系统评价分析
1. Beyond User Self-Reported Likert Scale Ratings: A Comparison Model for Automatic Dialog Evaluation
2. Dialogue Coherence Assessment Without Explicit Dialogue Act Labels
3. Don’t Say That! Making Inconsistent Dialogue Unlikely with Unlikelihood Training
4. Crawling and Preprocessing Mailing Lists At Scale for Dialog Analysis
5. Designing Precise and Robust Dialogue Response Evaluators
6. Evaluating Dialogue Generation Systems via Response Selection
7. Learning an Unreferenced Metric for Online Dialogue Evaluation
## 多模态对话
1. History for Visual Dialog: Do we really need it?
2. The Dialogue Dodecathlon: Open-Domain Knowledge and Image Grounded Conversational Agents
3. Towards Emotion-aided Multi-modal Dialogue Act Classification
4. Video-Grounded Dialogues with Pretrained Generation Language Models
## 对话新任务/数据
1. KdConv: A Chinese Multi-domain Dialogue Dataset Towards Multi-turn Knowledge-driven Conversation
2. More Diverse Dialogue Datasets via Diversity-Informed Data Collection
3. MuTual: A Dataset for Multi-Turn Dialogue Reasoning
4. Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset
5. ChartDialogs: Plotting from Natural Language Instructions
6. MIE: A Medical Information Extractor towards Medical Dialogues
7. Towards Conversational Recommendation over Multi-Type Dialogs
## 其他
1. Learning to execute instructions in a Minecraft dialogue

## 参考
1. [ACL 2020对话系统相关论文整理](https://bbs.huaweicloud.com/blogs/171902)

