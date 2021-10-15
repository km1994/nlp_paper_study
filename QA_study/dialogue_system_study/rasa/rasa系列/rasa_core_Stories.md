# 【关于 rasa -> Core -> Stories 】那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 

## 目录

- [【关于 rasa -> Core -> Stories 】那些你不知道的事](#关于-rasa---core---stories-那些你不知道的事)
  - [目录](#目录)
  - [一、概况图](#一概况图)
  - [二、动机](#二动机)
  - [三、什么是 Stories？](#三什么是-stories)
  - [四、Stories 主要分哪些部分呢？](#四stories-主要分哪些部分呢)
    - [4.1 用户输入 (User Messages)](#41-用户输入-user-messages)
    - [4.2 动作 （Action）](#42-动作-action)
    - [4.3 事件（Action）](#43-事件action)
      - [（1）Slot Events](#1slot-events)
      - [（2）Form Events](#2form-events)
  - [参考资料](#参考资料)


## 一、概况图

![](img/20200922212431.png)

## 二、动机

在对话管理（DM）中需要通过学习 获取到 必要的知识。那么，DM 的 训练数据从何而来？训练数据的格式是怎么样的？

- 答案： Stories

## 三、什么是 Stories？

- 介绍：Rasa 采用 Stories 作为一种训练 DM 模型的数据格式；
- 存放地址：该训练数据存放在 story.md 文件中。
- 说明：Stories 是用户和人工智能助手之间的对话的表示，转换为特定的格式，其中用户输入表示为相应的意图(和必要的实体)，而助手的响应表示为相应的操作名称。
- 该数据的格式如下所示：

```s
<!-- ##表示story的描述，没有实际作用 -->
## greet + location/price + cuisine + num people
* greet
   - utter_greet
* inform{"location": "rome", "price": "cheap"}
   - action_on_it
   - action_ask_cuisine
* inform{"cuisine": "spanish"}
   - action_ask_numpeople    
* inform{"people": "six"}
   - action_ack_dosearch
  
<!-- Form Action-->
## happy path 
* request_weather
   - weather_form
   - form{"name": "weather_form"}
   - form{"name": null}
```

## 四、Stories 主要分哪些部分呢？

### 4.1 用户输入 (User Messages)

- “*”：“*” 对应的信息 为 用户输入消息，采用 NLU 管道输出的 intent 和 entities 来表示可能的输入，policies 根据 intent 和 entities 预测下一步 action；
- 类别：
  - “* greet”：用户输入无 entity 的场景；
  - “* inform{"people": "six"}”：用户输入包含 entity 的场景，响应这一类 intent 为 普通 action;
  - “* request_weather”：用户输入Message对应的intent为form action情况；

### 4.2 动作 （Action）

- “-”：要执行动作(Action)；
- 分类：
  - utterance actions：在domain.yaml中定义以utter_为前缀，比如名为greet的意图，它的回复应为utter_greet；
  - custom actions：自定义动作，具体逻辑由我们自己实现，虽然在定义action名称的时候没有限制，但是还是建议以action_为前缀，比如名为inform的意图fetch_profile的意图，它的response可为action_fetch_profile；

### 4.3 事件（Action）

- “-”：要执行事件（Action）；
- 分类：
  - 槽值设置(SlotSet)：
  - 激活/注销表单(Form)：

#### （1）Slot Events

- 作用：当我们在自定义Action中设置了某个槽值，那么我们就需要在Story中Action执行之后显著的将这个SlotSet事件标注出来，格式为- slot{"slot_name": "value"}。
- 举例：我们在action_fetch_profile中设置了Slot名为account_type的值，代码如下：

```python
  from rasa_sdk.actions import Action
  from rasa_sdk.events import SlotSet
  import requests

  class FetchProfileAction(Action):
      def name(self):
          return "fetch_profile"

      def run(self, dispatcher, tracker, domain):
          url = "http://myprofileurl.com"
          data = requests.get(url).json
          return [SlotSet("account_type", data["account_type"])]
```

> 注：需要在Story中执行action_fetch_profile之后，添加- slot{"account_type" : "premium"}。

```s
## fetch_profile
* fetch_profile
   - action_fetch_profile
   - slot{"account_type" : "premium"}
   - utter_welcome_premium
```

> 如果您的自定义Action中将槽值重置为None，则对应的事件为-slot{"slot_name": null}

#### （2）Form Events

- 三种形式的表单事件(Form Events):
  - Form Action事件:
    - 介绍：表单动作事件，是自定义Action的一种，用于一个表单操作
    - “- restaurant_form”
  - Form activation事件：
    - 介绍：激活表单事件，当form action事件执行后，会立马执行该事件
    - “- form{"name": "restaurant_form"}”
  - Form deactivation事件:
    - 介绍：注销表单事件，作用与form activation相反
    - “- form{"name": null}”

```
## happy path
* request_weather
    - weather_form
    - form{"name": "weather_form"}
    - form{"name": null}
```



## 参考资料

1. [rasa 文档](https://rasa.com/docs/rasa/)
2. [Rasa中文聊天机器人开发指南(3)：Core篇](https://jiangdg.blog.csdn.net/article/details/105434136)【强烈推荐，小白入门经典】



