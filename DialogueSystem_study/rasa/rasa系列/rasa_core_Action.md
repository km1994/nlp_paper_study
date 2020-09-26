# 【关于 rasa -> Core -> Action 】那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 

## 目录

- [【关于 rasa -> Core -> Action 】那些你不知道的事](#关于-rasa---core---action-那些你不知道的事)
  - [目录](#目录)
  - [一、概况图](#一概况图)
  - [二、动机](#二动机)
  - [三、什么 是 action?](#三什么-是-action)
  - [四、action 有哪些类别？](#四action-有哪些类别)
    - [4.1 default actions](#41-default-actions)
    - [4.2 utter actions](#42-utter-actions)
    - [4.3 custom actions](#43-custom-actions)
  - [参考资料](#参考资料)

## 一、概况图

![](img/20200922212431.png)

## 二、动机

当Rasa NLU识别到用户输入Message的意图后，Rasa Core对话管理模块将如何对其作出回应呢？

> 答案：action

## 三、什么 是 action?

说白了就是 针对 Rasa NLU识别到用户输入Message的意图，Rasa Core对话管理模块 所 做成的 反应。

## 四、action 有哪些类别？

- 类别：
  - default actions
  - utter actions
  - custom actions

### 4.1 default actions

- 介绍：Rasa Core默认的一组actions，我们无需定义它们，直接可以story和domain中使用；
- 类别：
  - action_listen：监听action，Rasa Core在会话过程中通常会自动调用该action；
  - action_restart：重置状态，比初始化Slots(插槽)的值等；
  - action_default_fallback：当Rasa Core得到的置信度低于设置的阈值时，默认执行该action；

### 4.2 utter actions

- 介绍：以utter_为开头，仅仅用于向用户发送一条消息作为反馈的一类actions；
- 定义：只需要在domain.yml文件中的actions:字段定义以utter_为开头的action即可，而具体回复内容将被定义在templates:部分；

```s
actions:
  - utter_answer_greet
  - utter_answer_goodbye
  - utter_answer_thanks
  - utter_introduce_self
  - utter_introduce_selfcando
  - utter_introduce_selffrom
```

### 4.3 custom actions

- 介绍：CustomAction，即自定义action，允许开发者执行任何操作并反馈给用户，比如简单的返回一串字符串，或者控制家电、检查银行账户余额等等；
- Vs DefaultAction：
  - DefaultAction：需要我们在domain.yml文件中的actions部分先进行定义，然后在指定的webserver中实现它，其中，这个webserver的url地址在endpoint.yml文件中指定，并且这个webserver可以通过任何语言实现，当然这里首先推荐python来做，毕竟Rasa Core为我们封装好了一个rasa-core-sdk专门用来处理自定义action。


## 参考资料

1. [rasa 文档](https://rasa.com/docs/rasa/)
2. [Rasa中文聊天机器人开发指南(3)：Core篇](https://jiangdg.blog.csdn.net/article/details/105434136)【强烈推荐，小白入门经典】



