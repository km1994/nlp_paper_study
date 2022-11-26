# 【关于 FastSpeech2】那些你不知道的事

> 论文名称：FastSpeech 2: Fast and High-Quality End-to-End Text-to-Speech
> 
> 论文地址：https://arxiv.org/abs/2006.04558v1
> 
> github 地址：https://github.com/ming024/FastSpeech2

## 一、论文动机

非自回归TTS模型FastSpeech能够并行地合成语音，使合成速度显著提高。

但存在几个问题：

1. teacher-student的蒸馏过程非常复杂并且耗时；
2. 从teacher模型attention中提取的时长预测不够准确；
3. 用teacher模型预测的mel谱作为target，相比GT有信息损失从而导致结果音质受损。

## 二、论文方法

本文提出FastSpeech 2，能够通过以下方式很好解决TTS中的one-to-many映射问题：

1. 直接用GT的mel谱来训练模型，代替teacher模型输出；
2. 引入更具有变化的信息（pitch，energy，duration等）作为输入condition，即从语音中提取duration、pitch、energy，训练时用提取结果、inference时用text的预测结果。更进一步，设计了一个完全的E2E系模型FastSpeech 2s，直接从text并行地生成waveform

## 三、模型架构

![](img/微信截图_20221127011414.png)

1. encoder将phoneme embedding转换成phoneme hidden seq;
2. 然后设计了variance adaptor引入不同的声学特征信息;
3. 最终decoder将adapted hidden seq并行地转换成mel谱。

## 四、vs FastSpeech

- 相同点：
  - encoder、decoder主体使用的是前馈Transformer block（自注意+1D卷积）。
- 不同点：
  - FastSpeech 2不依靠teacher-student的蒸馏操作：直接用GT mel谱作为训练目标，可以避免蒸馏过程中的信息损失同时提高音质上限。
  - variance adaptor包括duration、pitch、energy的预测器predictor，其中DP通过训练数据中提取的强制对齐获得时长信息，这比从自回归teacher模型中提取更准确。


## 四、结论

1. FastSpeech 2 达到了FastSpeech训练速度的3倍，FastSpeech 2s具有更快的inference速度
2. FastSpeech 2和2s都比FastSpeech音质要好，FastSpeech 2甚至超过自回归模型

## 参考

1. [FastSpeech2——快速高质量语音合成](https://zhuanlan.zhihu.com/p/363808377)
2. [通过FastSpeech2中文合成项目梳理TTS流程2: 数据训练（train.py)](https://blog.csdn.net/weixin_42745601/article/details/120388860)
3. [FastSpeech2代码理解之模型实现（一）](https://blog.csdn.net/cool_numb/article/details/123992756)
4. [语音合成模型Fastspeech2技术报告](http://www.panjiangtao.cn/posts/Fastspeech2/)

