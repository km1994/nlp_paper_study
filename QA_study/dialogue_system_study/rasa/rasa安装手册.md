# 【关于 rasa 安装 】那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 

## 目录

- [【关于 rasa 安装 】那些你不知道的事](#关于-rasa-安装-那些你不知道的事)
  - [目录](#目录)
  - [安装 Rasa](#安装-rasa)
    - [Rasa 推荐 安装方式](#rasa-推荐-安装方式)
    - [sklearn  和 MITIE 库 安装](#sklearn-和-mitie-库-安装)
  - [项目初尝试](#项目初尝试)
    - [创建新项目](#创建新项目)
    - [NLU 训练数据介绍](#nlu-训练数据介绍)
    - [模型配置 介绍](#模型配置-介绍)
    - [写下你的第一个故事](#写下你的第一个故事)
    - [定义域](#定义域)
    - [模型训练](#模型训练)
    - [测试](#测试)
  - [Rasa 命令行 备忘录](#rasa-命令行-备忘录)
  - [Rasa 架构](#rasa-架构)
  - [参考资料](#参考资料)


## 安装 Rasa 

> 温馨提示：由于 安装 Rasa 过程中，会安装各种 乱七八糟的 依赖库（eg：tensorflow 2.0，...），导致 安装失败，所以建议 用 conda ，新建 一个 conda 环境，然后在 该环境上面开发。

- 创建环境
```
  $ conda create -n rasa python=3.6
```
- 激活环境
```
  $conda activate rasa
```


### Rasa 推荐 安装方式

```python
    pip install rasa-x --extra-index-url https://pypi.rasa.com/simple
```

> 注：该命令将同时安装 Rasa 和 Rasa X，如果你不想 安装 Rasa X，你可以用以下 命令：

```python
    pip install Rasa
```

### sklearn  和 MITIE 库 安装

```shell
  pip install -U scikit-learn sklearn-crfsuite
  pip install git+https://github.com/mit-nlp/MITIE.git
```

> 注：MITIE 库比较大，所以这种 安装方式容易出现问题，所以我用另一种安装方式

```shell
  $ git clone https://github.com/mit-nlp/MITIE.git
  $ cd MITIE/
  $ python setup.py install
```

安装结果

```shell
  Compiling src/text_feature_extraction.cpp
  Compiling ../dlib/dlib/threads/multithreaded_object_extension.cpp
  Compiling ../dlib/dlib/threads/threaded_object_extension.cpp
  Compiling ../dlib/dlib/threads/threads_kernel_1.cpp
  Compiling ../dlib/dlib/threads/threads_kernel_2.cpp
  Compiling ../dlib/dlib/threads/threads_kernel_shared.cpp
  Compiling ../dlib/dlib/threads/thread_pool_extension.cpp
  Compiling ../dlib/dlib/misc_api/misc_api_kernel_1.cpp
  Compiling ../dlib/dlib/misc_api/misc_api_kernel_2.cpp
  Linking libmitie.so
  Making libmitie.a
  Build Complete
  make[1]: Leaving directory `/web/workspace/yangkm/python_wp/nlu/DSWp/MITIE/mitielib'
  running build_py
  creating build
  creating build/lib
  creating build/lib/mitie
  copying mitielib/__init__.py -> build/lib/mitie
  copying mitielib/mitie.py -> build/lib/mitie
  copying mitielib/libmitie.so -> build/lib/mitie
  running install_lib
  copying build/lib/mitie/__init__.py -> /home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/mitie
  copying build/lib/mitie/mitie.py -> /home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/mitie
  copying build/lib/mitie/libmitie.so -> /home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/mitie
  byte-compiling /home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/mitie/__init__.py to __init__.cpython-36.pyc
  byte-compiling /home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/mitie/mitie.py to mitie.cpython-36.pyc
  running install_egg_info
  Writing /home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/mitie-0.7.0-py3.6.egg-info
```
> 注：会存在 一些 warning 警告，对结果 影响不大


## 项目初尝试

### 创建新项目

1. 第一步是创建一个新的Rasa项目。要做到这一点，运行下面的代码:
  
- 启动 Rasa 
  
```python
    rasa init --no-prompt
```

> 注：rasa init命令创建rasa项目所需的所有文件，并根据一些示例数据训练一个简单的机器人。如果你省略了——no-prompt参数，将会询问你一些关于项目设置的问题。

- 运行过程

```shell
$ rasa init --no-prompt
Welcome to Rasa! 🤖

To get started quickly, an initial project will be created.
If you need some help, check out the documentation at https://rasa.com/docs/rasa.

Created project directory at '/web/workspace/yangkm/python_wp/nlu/DSWp'.
Finished creating project structure.
Training an initial model...
Training Core model...
Processed Story Blocks: 100%|█████████████████████████████████████████████| 5/5 [00:00<00:00, 3562.34it/s, # trackers=1]
Processed Story Blocks: 100%|█████████████████████████████████████████████| 5/5 [00:00<00:00, 1523.54it/s, # trackers=5]
Processed Story Blocks: 100%|█████████████████████████████████████████████| 5/5 [00:00<00:00, 380.28it/s, # trackers=20]
Processed Story Blocks: 100%|█████████████████████████████████████████████| 5/5 [00:00<00:00, 301.26it/s, # trackers=24]
Processed trackers: 100%|█████████████████████████████████████████████████| 5/5 [00:00<00:00, 2233.39it/s, # actions=16]
Processed actions: 16it [00:00, 14986.35it/s, # examples=16]
Processed trackers: 100%|█████████████████████████████████████████████| 231/231 [00:00<00:00, 899.80it/s, # actions=126]
Epochs:   0%|                                                                                   | 0/100 [00:00<?, ?it/s]/home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/rasa/utils/tensorflow/model_data.py:386: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
  final_data[k].append(np.concatenate(np.array(v)))
Epochs: 100%|████████████████████████████████████| 100/100 [00:06<00:00, 14.77it/s, t_loss=0.083, loss=0.009, acc=1.000]
2020-09-17 16:46:48 INFO     rasa.utils.tensorflow.models  - Finished training.
2020-09-17 16:46:48 INFO     rasa.core.agent  - Persisted model to '/tmp/tmpjkpkgun2/core'
Core model training completed.
Training NLU model...
2020-09-17 16:46:48 INFO     rasa.nlu.training_data.training_data  - Training data stats:
2020-09-17 16:46:48 INFO     rasa.nlu.training_data.training_data  - Number of intent examples: 43 (7 distinct intents)
2020-09-17 16:46:48 INFO     rasa.nlu.training_data.training_data  -   Found intents: 'mood_unhappy', 'bot_challenge', 'deny', 'affirm', 'greet', 'mood_great', 'goodbye'
2020-09-17 16:46:48 INFO     rasa.nlu.training_data.training_data  - Number of response examples: 0 (0 distinct responses)
2020-09-17 16:46:48 INFO     rasa.nlu.training_data.training_data  - Number of entity examples: 0 (0 distinct entities)
2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component WhitespaceTokenizer
2020-09-17 16:46:48 INFO     rasa.nlu.model  - Finished training component.
2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component RegexFeaturizer
2020-09-17 16:46:48 INFO     rasa.nlu.model  - Finished training component.
2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component LexicalSyntacticFeaturizer
2020-09-17 16:46:48 INFO     rasa.nlu.model  - Finished training component.
2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component CountVectorsFeaturizer
2020-09-17 16:46:48 INFO     rasa.nlu.model  - Finished training component.
2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component CountVectorsFeaturizer
2020-09-17 16:46:48 INFO     rasa.nlu.model  - Finished training component.
2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component DIETClassifier
/home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/rasa/utils/common.py:363: UserWarning: You specified 'DIET' to train entities, but no entities are present in the training data. Skip training of entities.
Epochs:   0%|                                                                                   | 0/100 [00:00<?, ?it/s]/home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/rasa/utils/tensorflow/model_data.py:386: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
  final_data[k].append(np.concatenate(np.array(v)))
Epochs: 100%|████████████████████████████████| 100/100 [00:05<00:00, 18.36it/s, t_loss=1.475, i_loss=0.095, i_acc=1.000]
2020-09-17 16:46:58 INFO     rasa.utils.tensorflow.models  - Finished training.
2020-09-17 16:46:59 INFO     rasa.nlu.model  - Finished training component.
2020-09-17 16:46:59 INFO     rasa.nlu.model  - Starting to train component EntitySynonymMapper
2020-09-17 16:46:59 INFO     rasa.nlu.model  - Finished training component.
2020-09-17 16:46:59 INFO     rasa.nlu.model  - Starting to train component ResponseSelector
2020-09-17 16:46:59 INFO     rasa.nlu.selectors.response_selector  - Retrieval intent parameter was left to its default value. This response selector will be trained on training examples combining all retrieval intents.
2020-09-17 16:46:59 INFO     rasa.nlu.model  - Finished training component.
2020-09-17 16:46:59 INFO     rasa.nlu.model  - Successfully saved model into '/tmp/tmpjkpkgun2/nlu'
NLU model training completed.
Your Rasa model is trained and saved at '/web/workspace/yangkm/python_wp/nlu/DSWp/models/20200917-164632.tar.gz'.
If you want to speak to the assistant, run 'rasa shell' at any time inside the project directory.
```

- 运行结果

将在该目录下创建以下文件：

<table>
    <thead>
        <td>文件名称</td><td>作用说明</td>
    </thead>
    <tr>
        <td>init.py</td><td>帮助python查找操作的空文件</td>
    </tr>
    <tr>
        <td>actions.py</td><td>为你的自定义操作编写代码</td>
    </tr>
    <tr>
        <td>config.yml ‘*’</td><td>配置NLU和Core模型</td>
    </tr>
    <tr>
        <td>credentials.yml</td><td>连接到其他服务的详细信息</td>
    </tr>
    <tr>
        <td>data/nlu.md ‘*’</td><td>你的NLU训练数据</td>
    </tr>
    <tr>
        <td>data/stories.md ‘*’</td><td>你的故事</td>
    </tr>
    <tr>
        <td>domain.yml ‘*’</td><td>你的助手的域</td>
    </tr>
    <tr>
        <td>endpoints.yml</td><td>接到fb messenger等通道的详细信息</td>
    </tr>
    <tr>
        <td>models/.tar.gz</td><td>你的初始模型</td>
    </tr>
</table>

> 注：最重要的文件用“*”标记。你将在本教程中了解所有这些文件。

### NLU 训练数据介绍

想让 NLU 理解用户消息，需要将 该消息 转化为 结构化 数据，数据格式如下所示：

```shell
  $ cat data/nlu.md
  ## intent:greet
  - hey
  - hello
  - hi
  - good morning
  - good evening
  - hey there

  ## intent:goodbye
  - bye
  - goodbye
  - see you around
  - see you later

  ## intent:affirm
  - yes
  - indeed
  - of course
  - that sounds good
  - correct

  ## intent:deny
  - no
  - never
  - I don't think so
  - don't like that
  - no way
  - not really

  ## intent:mood_great
  - perfect
  - very good
  - great
  - amazing
  - wonderful
  - I am feeling very good
  - I am great
  - I'm good

  ## intent:mood_unhappy
  - sad
  - very sad
  - unhappy
  - bad
  - very bad
  - awful
  - terrible
  - not very good
  - extremely sad
  - so sad

  ## intent:bot_challenge
  - are you a bot?
  - are you a human?
  - am I talking to a bot?
  - am I talking to a human?
```

> 注：
> 
> “##” 开始的行定义意图的名称 
> “-” 表示 意图所对应的 关键词

### 模型配置 介绍

```shell
$ cat config.yml 
# Configuration for Rasa NLU.
# https://rasa.com/docs/rasa/nlu/components/
language: en
pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100

# Configuration for Rasa Core.
# https://rasa.com/docs/rasa/core/policies/
policies:
  - name: MemoizationPolicy
  - name: TEDPolicy
    max_history: 5
    epochs: 100
  - name: MappingPolicy
```

> 注：language 和 pipeline 键指定应该如何构建 NLU 模型。policies 键定义 Core 模型将使用的策略。

### 写下你的第一个故事

在这个阶段，你将教会你的助手如何回复你的信息。这称为对话管理(dialogue management)，由你的Core模型来处理。

Core模型以训练“故事”的形式从真实的会话数据中学习。故事是用户和助手之间的真实对话。带有意图和实体的行反映了用户的输入和操作名称，操作名称展示了助手应该如何响应。

下面是一个简单对话的例子。用户说你好，助手也说你好。故事是这样的:

```shell
$ cat stories.md 
## happy path
* greet
  - utter_greet
* mood_great
  - utter_happy

## sad path 1
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* affirm
  - utter_happy

## sad path 2
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* deny
  - utter_goodbye

## say goodbye
* goodbye
  - utter_goodbye

## bot challenge
* bot_challenge
  - utter_iamabot

```

你可以在故事中看到完整的细节。

以-开头的行是助手所采取的操作。在本教程中，我们所有的操作都是发送回用户的消息，比如utter_greet，但是一般来说，一个操作可以做任何事情，包括调用API和与外部世界交互。

运行下面的命令查看文件data/stories.md中的示例故事:

```
  cat data/stories.md
```

### 定义域

域定义了助手所处的环境:它应该期望得到什么用户输入、它应该能够预测什么操作、如何响应以及存储什么信息。我们助手的域名保存在一个名为domain.yml的文件中:

```shell
$ cat domain.yml 
intents:
  - greet
  - goodbye
  - affirm
  - deny
  - mood_great
  - mood_unhappy
  - bot_challenge

responses:
  utter_greet:
  - text: "Hey! How are you?"

  utter_cheer_up:
  - text: "Here is something to cheer you up:"
    image: "https://i.imgur.com/nGF1K8f.jpg"

  utter_did_that_help:
  - text: "Did that help you?"

  utter_happy:
  - text: "Great, carry on!"

  utter_goodbye:
  - text: "Bye"

  utter_iamabot:
  - text: "I am a bot, powered by Rasa."

session_config:
  session_expiration_time: 60
  carry_over_slots_to_new_session: true

```
<table>
  <thead>
      <td></td><td>解释说明</td>
  </thead>
  <tr>
      <td>intents</td><td>你希望用户说的话</td>
  </tr>
  <tr>
      <td>actions</td><td>你的助手能做的和能说的</td>
  </tr>
  <tr>
      <td>templates</td><td>你的助手可以说的东西的模板字符串</td>
  </tr>
</table>

- Rasa Core 工作机制：
  - 在对话的每个步骤中选择正确的操作来执行。在本例中，我们的操作只是向用户发送一条消息。这些简单的话语操作是从域中以utter_开头的操作。助手将根据templates部分中的模板返回一条消息。

### 模型训练

每当我们添加新的NLU或Core数据，或更新域或配置时，我们都需要根据示例故事和NLU数据重新训练一个神经网络。为此，运行下面的命令。该命令将调用Rasa Core和NLU训练函数，并将训练后的模型存储到models/目录中。该命令只会在数据或配置发生更改时自动对不同的模型部件进行重新训练。

```
  rasa train
  echo "Finished training."
```

rasa train命令将同时查找NLU和Core数据，并训练一个组合模型。

### 测试

恭喜你! 🚀 你刚刚建立了一个完全由机器学习驱动的助手。 下一步就是尝试一下!如果你正在本地机器上学习本教程，请运行以下命令与助手对话：

```shell
  
  $ rasa shell
  2020-09-17 19:44:19 INFO     root  - Connecting to channel 'cmdline' which was specified by the '--connector' argument. Any other channels will be ignored. To connect to all given channels, omit the '--connector' argument.
  2020-09-17 19:44:19 INFO     root  - Starting Rasa server on http://localhost:5005
  2020-09-17 19:44:26 INFO     root  - Rasa server is up and running.
  Bot loaded. Type a message and press enter (use '/stop' to exit): 
  Your input ->  hello                                                                                                    
  Hey! How are you?
  Your input ->  i'm fine                                                                                                 
  Great, carry on!
  Your input ->  Did that help you                                                                                        
  I am a bot, powered by Rasa.
  Your input ->  are you a human?                                                                                         
  I am a bot, powered by Rasa.
  Your input ->  am I talking to a bot?                                                                                   
  I am a bot, powered by Rasa.
  Your input ->  see you around                                                                                           
  Bye
  Your input ->  of course                                                                                                
  Great, carry on!
  Your input ->  I don't think so                                                                                         
  Bye
  Your input ->  no way                                                                                                   
  Bye
  Your input ->  I am feeling very good                                                                                   
  Great, carry on!
  Your input ->  very bad                                                                                                 
  Here is something to cheer you up:
  Image: https://i.imgur.com/nGF1K8f.jpg
  Did that help you?
  Your input ->  are you a bot?                                                                                           
  I am a bot, powered by Rasa.
```

## Rasa 命令行 备忘录

<table>
  <thead>
      <td>命令</td><td>作用说明</td>
  </thead>
  <tr>
      <td>rasa init</td><td>使用示例训练数据，操作和配置文件创建新项目</td>
  </tr>
  <tr>
      <td>rasa train</td><td>使用你的NLU数据和故事训练模型，在./model中保存训练的模型</td>
  </tr>
  <tr>
      <td>rasa interactive</td><td>启动交互式学习会话，通过聊天创建新的训练数据</td>
  </tr>
  <tr>
      <td>rasa shell</td><td>加载已训练的模型，并让你在命令行上与助手交谈</td>
  </tr>
  <tr>
      <td>rasa run</td><td>使用已训练的的模型启动Rasa服务。有关详细信息，请参阅运行服务文档</td>
  </tr>
  <tr>
      <td>rasa run actions</td><td>使用Rasa SDK启动操作服务</td>
  </tr>
  <tr>
      <td>rasa visualize</td><td>可视化故事</td>
  </tr>
  <tr>
      <td>rasa test</td><td>使用你的测试NLU数据和故事测试已训练的Rasa模型</td>
  </tr>

  <tr>
      <td>rasa data split nlu</td><td>根据指定的百分比执行NLU数据的拆分</td>
  </tr>
  <tr>
      <td>rasa data convert nlu</td><td>在不同格式之间转换NLU训练数据</td>
  </tr>
  <tr>
      <td>rasa x</td><td>在本地启动Rasa X</td>
  </tr>
  <tr>
      <td>rasa -h</td><td>显示所有可用命令</td>
  </tr>
</table>

具体介绍，可以查看 [Rasa 命令行界面](http://rasachatbot.com/3_Command_Line_Interface/)

## Rasa 架构 

![](img/20200918105542.png)

- Rasa构建的助手如何响应消息的基本步骤：
  - 1. 收到消息并将其传递给解释器(Interpreter)，解释器将其转换为包含原始文本，意图和找到的任何实体的字典。这部分由NLU处理;
  - 2. 跟踪器(Tracker)是跟踪对话状态的对象。它接收新消息进入的信息;
  - 3. 策略(Policy)接收跟踪器的当前状态。 该策略选择接下来采取的操作(action)。 
  - 4. 选择的操作由跟踪器记录。 
  - 5. 响应被发送给用户。


## 参考资料

1. [Rasa 安装](http://rasachatbot.com/2_Rasa_Tutorial/#rasa)
2. [Rasa 聊天机器人中文官方文档|磐创AI](http://rasachatbot.com/)
3. [Rasa 学习](https://blog.csdn.net/ljp1919/category_9656007.html)
4. [rasa_chatbot_cn](https://github.com/GaoQ1/rasa_chatbot_cn)
5. [用Rasa NLU构建自己的中文NLU系统](http://www.crownpku.com/2017/07/27/用Rasa_NLU构建自己的中文NLU系统.html)
6. [Rasa_NLU_Chi](https://github.com/crownpku/Rasa_NLU_Chi)
7. [_rasa_chatbot](https://github.com/zqhZY/_rasa_chatbot)
8. [rasa 源码分析](https://www.zhihu.com/people/martis777/posts)
