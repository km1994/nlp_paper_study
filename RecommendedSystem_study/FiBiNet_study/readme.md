# 【关于 FiBiNET】那些你不知道的事

> 笔者：杨夕
> 
> 论文：FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction（结合特征重要性和双线性特征相互作用进行点击率预测）
>
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 摘要

### 广告和提要排名（Advertising and feed ranking ）应用场景
Advertising and feed ranking are essential to many Internet companies such as Facebook and Sina Weibo. 

- 应用领域
  - Among many real-world advertising and feed ranking systems；
  - click through rate (CTR) prediction plays a central role. 

### 常用方法

- 方法：
  - logistic regression； 
  - tree based models；
  - factorization machine based models；
  -  deep learning based CTR models. 

### 存在问题

However, many current works calculate the feature interactions in a simple way such as **Hadamard product** and **inner product** and **they care less about the importance of features**. 

### 论文方法

In this paper, a new model named **FiBiNET** as an abbreviation for Feature Importance and Bilinear feature Interaction NETwork is proposed to **dynamically learn the feature importance and fine-grained feature interactions**. 

- 特点
  - the FiBiNET can dynamically learn the importance of features via the Squeeze-Excitation network (SENET) mechanism; 
  - it is able to effectively learn the feature interactions via bilinear function. 

### 实验结果

We conduct extensive experiments on two realworld datasets and show that our shallow model outperforms other shallow models such as factorization machine(FM) and field-aware factorization machine(FFM). In order to improve performance further, we combine a classical deep neural network(DNN) component with the shallow model to be a deep model. The deep FiBiNET consistently outperforms the other state-of-the-art deep models such as DeepFM and extreme deep factorization machine(XdeepFM).

## 引言


