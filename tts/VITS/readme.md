# 【关于 VITS】那些你不知道的事


2023年02月27日 端到端音调可控TTS的无基频变音调推理

论文：PITS: Variational Pitch Inference without Fundamental Frequency for End-to-End Pitch-controllable TTS

机构：VITS原团队，韩国科学院

代码：https://github.com/anonymous-pits/pits

目的：PITS在VITS的基础上，结合了Yingram编码器、Yingram解码器和对抗式的移频合成训练来实现基音可控性。

图片

VITS歌声转换innnky/so-vits-svc，具备破万star潜力：

https://github.com/innnky/so-vits-svc

效果最好开源PlayVoice/vits_chinese，提供预训练模型：

https://github.com/PlayVoice/vits_chinese

2021 年 6 月 11 日 VITS 论文和代码发布：

论文：Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

代码：github.com/jaywalnut310

机构：韩国科学院

会议：ICML 2021

作者其他论文：HiFiGAN、GlowTTS



2021 年 6 月 21 日 与 VITS 同架构论文：

论文：Glow-WaveGAN：Learning Speech Representations from GAN-based Variational Auto-Encoder For High Fidelity Flow-based Speech Synthesis

机构：西北工业大学，腾讯 AI 实验室

会议：INTERSPEECH 2021



2021 年 10 月 15 日 VITS 评估论文发布：

论文：ESPnet2-TTS Extending the Edge of TTS Research

代码：github.com/espnet/espne

机构：开源机构 ESPnet、卡梅隆大学、东京大学等

目的：对先进的语音合成系统进行评估，尤其是 VITS；ESPnet 提供的 152 个预训练模型（ASR+TTS）中有 48 为 VITS 语音合成模型。



2021 年 10 月 17 日 VITS 相关论文：

论文：VISinger: Variational Inference with Adversarial Learning for End-to-End Singing Voice Synthesis

机构：西北工业大学、网易伏羲 AI 实验室

目的：基于 VITS 实现的歌声合成系统



2021 年 12 月 4 日 VITS 相关论文：

论文：YourTTS：Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone

代码：edresson.github.io/Your

机构：开源机构 coqui-ai/TTS

目的：基于 VITS 实现跨语言语音合成和声音转换



2021 年 12 月 23 日 语音合成专题学术论坛：

机构：CCF 语音对话与听觉专委会

在会议中，微软亚洲研究院主管研究员谭旭博士，透露基于 VITS 实现的构建录音水平的文本到语音合成系统：DelightfulTTS 2 (Blizzard Challenge 2021/Ongoing)，论文还未公开



2022年3月30日 VoiceMe：TTS中的个性化语音生成

论文：VoiceMe: Personalized voice generation in TTS

代码：github.com/polvanrijn/V

机构：University of Cambridge etc

目的：使用来自最先进的说话人验证模型（SpeakerNet）的说话人嵌入来调节TTS模型。展示了用户可以创建与人脸、艺术肖像和卡通照片非常匹配的声音；使用wav2lip合成口型。



2022年3月30日 Nix-TTS：VITS模型的加速

论文：Nix-TTS: An Incredibly Lightweight End-to-End Text-to-Speech Model via Non End-to-End Distillation

代码：github.com/choiHkk/nix-

演示：github.com/rendchevi/ni

机构：Amazon (UK) etc

目的：使用VITS作为教师模型，使用Nix-TTS作为学生模型，大约得到3倍的加速



2022年5月10日 NaturalSpeech：具有人类水平质量的端到端文本到语音合成

论文：NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality

机构：Microsoft Research Asia & Microsoft Azure Speech Xu Tan

目的：通过几个关键设计来增强从文本到后文本的能力，降低从语音到后文本的复杂性，包括音素预训练、可微时长建模、双向前/后建模以及VAE中的记忆机制。



2022年6月2日 AdaVITS: Tiny VITS

论文：AdaVITS: Tiny VITS for Low Computing Resource Speaker Adaptation

机构：西工大&&腾讯

目的：用于低计算资源的说话人自适应；提出了一种基于iSTFT的波形构造解码器，以取代原VITS中资源消耗较大的基于上采样的解码器；引入了NanoFlow来共享流块之间的密度估计；将语音后验概率（PPG）用作语言特征；



2022年6月27日

论文：End-to-End Text-to-Speech Based on Latent Representation of Speaking Styles Using Spontaneous Dialogue

目的：语音上下文对话风格；两个阶段进行训练：第一阶段，训练变分自动编码器（VAE）-VITS，从语音中提取潜在说话风格表示的风格编码器与TTS联合训练。第二阶段，训练一个风格预测因子来预测从对话历史中综合出来的说话风格。以适合对话上下文的风格合成语音。



2022年6月27日 Sane-TTS

论文：SANE-TTS: Stable And Natural End-to-End Multilingual Text-to-Speech

机构：MINDsLab Inc，KAIST

目的：跨语言克隆；引入了说话人正则化丢失，在跨语言合成过程中提高了语音的自然度，并引入了域对抗训练。在持续时间预测器中用零向量代替说话人嵌入，稳定了跨语言推理。



2022年7月6日 Glow-WaveGAN 2

论文：Glow-WaveGAN 2: High-quality Zero-shot Text-to-speech Synthesis and Any-to-any Voice Conversion

演示：leiyi420.github.io/glow

机构：腾讯

目的：零资源语音克隆，任意到任意的变声；使用通用预训练大模型WaveGAN，替换VAE和HIFIGAN；



2022年7月14日 CLONE

论文：Controllable and Lossless Non-Autoregressive End-to-End Text-to-Speech

演示：xcmyz.github.io/CLONE/

机构：字节、清华

目的：【VITS cannot control prosody.】一对多映射问题；缺乏真实声学特征的监督；归一化流的变分自动编码器来建模语音中的潜在韵律信息；双并行自动编码器，在训练期间引入对真实声学特征的监督；



2022年7月 nix-tts

名称：End-To-End SpeechSynthesis system with knowledge distillation

代码：github.com/choiHkk/nix-

目的：vits知识蒸馏，模型压缩



2022年9月 interspeech_2022

论文：TriniTTS: Pitch-controllable End-to-end TTS without External Aligner

机构：现代汽车、卡梅伦

目的：VITS架构中添加基音控制；去掉Flow，加速；



2022年10月6日 无标注训练

论文：Transfer Learning Framework for Low-Resource Text-to-Speech using a Large-Scale Unlabeled Speech Corpus

代码：github.com/hcy71o/Trans

机构：三星等

目的：使用大规模无标注语料训练TTS；使用wav2vec2.0;



2022年10月28日 基于VITS架构的变声

论文：FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion

代码：github.com/olawod/freev

目的：本文采用了端到端的VITS框架来实现高质量的波形重构，并提出了无需文本标注的干净内容信息提取策略。通过在WavLM特征中引入信息瓶颈，对内容信息进行分解，并提出基于谱图大小调整的数据增强方法，以提高提取内容信息的纯度。



2022年10月31日 VITS加速

论文：Lightweight and High-Fidelity End-to-End Text-to-Speech with Multi-Band Generation and Inverse Short-Time Fourier Transform

代码：github.com/MasayaKawamu

机构：University of Tokyo, Japan,LINE Corp., Japan.

目的：比VITS快4.1倍，音质无影响；1）用简单的iSTFT部分地替换计算上最昂贵的卷积（2倍加速），2）PQMF的多频带生成来生成波形。



2022年10月31日 Period VITS情感TTS

论文：Period VITS: Variational Inference with Explicit Pitch Modeling for End-to-end Emotional Speech Synthesis

机构：University of Tokyo, Japan,LINE Corp., Japan.

目的：解码器中使用NSF，情感表达准确



2022年11月8日 VISinger 2

论文：VISinger 2: High-Fidelity End-to-End Singing Voice Synthesis Enhanced by Digital Signal Processing Synthesizer

机构：School of Computer Science, Northwestern Polytechnical University, Xi’an, China, DiDi Chuxing, Beijing, China

目的：NSF+VISinger



2023年1月 VITS onnx推理代码

代码：GitHub - rhasspy/larynx2: A fast, local neural text to speech system

机构：Rhasspy

目的：可导出onnx模型的VITS训练代码；C++推理代码；提供安装包，和预训练模型；支持平台 desktop Linux && Raspberry Pi 4；



2023年2月 VITS 变声 QuickVC

论文：QuickVC: Many-to-any Voice Conversion Using Inverse Short-time Fourier Transform for Faster Conversion

代码：github.com/quickvc/Quic

目的：SoftVC + VITS + iSTFT



2023年 wetts vits产品化

代码：GitHub - wenet-e2e/wetts: Production First and Production Ready End-to-End Text-to-Speech Toolkit

功能：前端处理，onnx，流式VITS？~



2024年02月27日 端到端音调可控TTS的无基频变音调推理

论文：PITS: Variational Pitch Inference without Fundamental Frequency for End-to-End Pitch-controllable TTS

机构：VITS团队

目的：PITS在VITS的基础上，结合了Yingram编码器、Yingram解码器和对抗式的移频合成训练来实现基音可控性。


## 参考

1. [2023.02.27重磅更新 举世无双语音合成系统 VITS 发展历程](https://mp.weixin.qq.com/s/AQRw4o3SkD6xH47rMUDRTQ)

