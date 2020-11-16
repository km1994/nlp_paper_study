# 【关于 DDPG】 那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 介绍

1. DDPG(Deep Deterministic Policy  Gradient)： 在连续控制领域经典的RL算法，是DQN在处理连续动作空间的一个扩充。具体地，从命名就可以看出，Deep是使用了神经网络；Deterministic 表示 DDPG 输出的是一个确定性的动作，可以用于连续动作的一个环境；Policy Gradient 代表的是它用到的是策略网络，并且每个 step 都会更新一次 policy 网络，也就是说它是一个单步更新的 policy 网络。其与DQN都有目标网络和经验回放的技巧，在经验回放部分是一致的，在目标网络的更新有些许不同。