# 【关于 Joint NER】那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 论文名称：Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy
> 
> 【注：手机阅读可能图片打不开！！！】
> 

## 摘要

Joint extraction of entities and relations aims to detect entity pairs along with their relations using a single model. 
实体和关系的联合提取旨在使用单个模型检测实体对及其关系。

Prior work typically solves this task in the extract-then-classify or unified labeling manner. 

However, these methods either suffer from the redundant entity pairs, or ignore the important inner structure in the process of extracting entities and relations. 

To address these limitations, in this paper, we first decompose the joint extraction task into two interrelated subtasks, namely HE extraction and TER extraction. The former subtask is to distinguish all head-entities that may be involved with target relations, and the latter is to identify corresponding tail-entities and relations for each extracted head-entity. Next, these two subtasks are further deconstructed into several sequence labeling problems based on our proposed span-based tagging scheme, which are conveniently solved by a hierarchical boundary tagger and a multi-span decoding algorithm. 

Owing to the reasonable decomposition strategy, our model can fully capture the semantic interdependency between different steps, as well as reduce noise from irrelevant entity pairs. Experimental results show that our method outperforms previous work by 5.2%, 5.9% and 21.5% (F1 score), achieving a new state-of-the-art on three public datasets.