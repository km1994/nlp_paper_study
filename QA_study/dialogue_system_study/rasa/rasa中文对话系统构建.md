# 【关于 rasa中文对话系统构建】那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 
> 学习项目：https://github.com/zqhZY/_rasa_chatbot

## 目录

- [【关于 rasa中文对话系统构建】那些你不知道的事](#关于-rasa中文对话系统构建那些你不知道的事)
  - [目录](#目录)
  - [一、安装 Rasa 内容](#一安装-rasa-内容)
    - [1.1 安装 中文版本 的 Rasa-nlu](#11-安装-中文版本-的-rasa-nlu)
    - [1.2 安装 rasa](#12-安装-rasa)
      - [安装过程中遇到的问题](#安装过程中遇到的问题)
    - [1.3 sklearn  和 MITIE 库 安装](#13-sklearn--和-mitie-库-安装)
    - [1.4 安装 rasa_core](#14-安装-rasa_core)
  - [二、新建项目](#二新建项目)
    - [2.1 安装rasa，自动生成rasa项目所需文件](#21-安装rasa自动生成rasa项目所需文件)
    - [2.2 测试助手](#22-测试助手)
  - [三、Rasa使用–构建简单聊天机器人](#三rasa使用构建简单聊天机器人)
    - [3.1 Rasa工作原理](#31-rasa工作原理)
    - [3.2 构建NLU样本](#32-构建nlu样本)
      - [3.2.1 data/nlu.md](#321-datanlumd)
    - [3.3 构建Core样本](#33-构建core样本)
      - [3.3.1 data/stories.md](#331-datastoriesmd)
      - [3.3.2 domain.yml](#332-domainyml)
    - [3.4 训练NLU和CORE模型](#34-训练nlu和core模型)
      - [3.4.1 config.yml](#341-configyml)
      - [3.4.2 模型训练](#342-模型训练)
  - [四、配置Http和Action](#四配置http和action)
    - [4.1 credentials.yml](#41-credentialsyml)
    - [4.2 endpoints.yml](#42-endpointsyml)
    - [4.3 action.py](#43-actionpy)
  - [五、启动对话](#五启动对话)
    - [5.1 通过 post 请求 查询](#51-通过-post-请求-查询)
      - [（1）启动Rasa服务](#1启动rasa服务)
      - [（2）启动Action服务](#2启动action服务)
      - [（3） 效果演示](#3-效果演示)
    - [5.2 聊天窗 式 对话](#52-聊天窗-式-对话)
      - [（一）启动Action服务](#一启动action服务)
      - [（二） 开启对话窗口](#二-开启对话窗口)
      - [（三）](#三)


## 一、安装 Rasa 内容

> 温馨提示：由于 安装 Rasa 过程中，会安装各种 乱七八糟的 依赖库（eg：tensorflow 2.0，...），导致 安装失败，所以建议 用 conda ，新建 一个 conda 环境，然后在 该环境上面开发。

### 1.1 安装 中文版本 的 Rasa-nlu 

```
  $ git clone https://github.com/crownpku/Rasa_NLU_Chi.git
  $ cd rasa_nlu
  $ pip install -r requirements.txt
  $ python setup.py install
```

### 1.2 安装 rasa 

```
  pip --default-timeout=500 install -U rasa
  pip install --upgrade --ignore-installed tensorflow

```

#### 安装过程中遇到的问题

- 问题一：
```
  ERROR: After October 2020 you may experience errors when installing or updating packages. This is because pip will change the way that it resolves dependency conflicts.

  We recommend you use --use-feature=2020-resolver to test your packages with the new resolver before it becomes the default.
```
然後使用者都要自己去解相依性問題 … pip 從 2020/10 月後會改用 2020-resolver 這個 resolver，pip 會自己去解決相依性問題，在官方還沒預設採用 2020-resolver 之前要用 --use-feature=2020-resolver 手動指定 resolver

假設升級 aws-cdk.aws-sns-subscriptions 則有一連串也需要被更新的套件也會被拉上來

1. 解决方法一
```
  python -m pip install --upgrade pip
  pip install --use-feature=2020-resolver --upgrade aws-cdk.aws-sns-subscriptions
```
然后重新 安装

2. 解决方法二

解决方法: 在pip命令中加入–use-feature=2020-resolver参数就可以了, 比如pip install xxx --use-feature=2020-resolver

### 1.3 sklearn  和 MITIE 库 安装

```shell
  pip install -U scikit-learn sklearn-crfsuite
  pip install git+https://github.com/mit-nlp/MITIE.git
```
- 下载 中文词向量 total_word_feature_extractor_zh.dat https://mega.nz/#!EWgTHSxR!NbTXDAuVHwwdP2-Ia8qG7No-JUsSbH5mNQSRDsjztSA

> 注：MITIE 库比较大，所以这种 安装方式容易出现问题，所以我用另一种安装方式

```s
  $ git clone https://github.com/mit-nlp/MITIE.git
  $ cd MITIE/
  $ python setup.py install
```

安装结果

```s
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


### 1.4 安装 rasa_core

```s
pip install rasa_core==0.9.0
```

> 注：

```
  # NameVersion   Build  Channel
  absl-py   0.9.0 <pip>
  aiofiles  0.5.0 <pip>
  aiohttp   3.6.2 <pip>
  APScheduler   3.6.3 <pip>
  astor 0.8.1 <pip>
  astunparse1.6.3 <pip>
  async-generator   1.10  <pip>
  async-timeout 3.0.1 <pip>
  attrs 19.3.0<pip>
  Automat   20.2.0<pip>
  bleach1.5.0 <pip>
  boto3 1.15.1<pip>
  botocore  1.18.1<pip>
  bz2file   0.98  <pip>
  cachetools4.1.1 <pip>
  certifi   2020.6.20 <pip>
  certifi   2016.2.28py36_0https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  cffi  1.14.3<pip>
  chardet   3.0.4 <pip>
  click 7.1.2 <pip>
  cloudpickle   1.3.0 <pip>
  colorclass2.2.0 <pip>
  coloredlogs   10.0  <pip>
  colorhash 1.0.2 <pip>
  ConfigArgParse1.2.3 <pip>
  constantly15.1.0<pip>
  contextvars   2.4   <pip>
  cryptography  3.1   <pip>
  cycler0.10.0<pip>
  dataclasses   0.7   <pip>
  decorator 4.4.2 <pip>
  dill  0.3.2 <pip>
  dnspython 1.16.0<pip>
  docopt0.6.2 <pip>
  docutils  0.16  <pip>
  dopamine-rl   3.1.7 <pip>
  fakeredis 0.10.3<pip>
  fbmessenger   6.0.0 <pip>
  Flask 1.1.2 <pip>
  Flask-Cors3.0.9 <pip>
  flax  0.2.1 <pip>
  future0.16.0<pip>
  gast  0.2.2 <pip>
  gevent1.5.0 <pip>
  gin-config0.3.0 <pip>
  google-api-core   1.22.2<pip>
  google-api-python-client  1.12.1<pip>
  google-auth   1.21.2<pip>
  google-auth-httplib2  0.0.4 <pip>
  google-auth-oauthlib  0.4.1 <pip>
  google-pasta  0.2.0 <pip>
  googleapis-common-protos  1.52.0<pip>
  GPUtil1.3.0 <pip>
  graphviz  0.8.4 <pip>
  greenlet  0.4.16<pip>
  grpcio1.32.0<pip>
  gunicorn  20.0.4<pip>
  gym   0.17.2<pip>
  h11   0.8.1 <pip>
  h23.2.0 <pip>
  h5py  2.10.0<pip>
  hpack 3.0.0 <pip>
  hstspreload   2020.9.15 <pip>
  html5lib  0.9999999 <pip>
  httpcore  0.3.0 <pip>
  httplib2  0.18.1<pip>
  httptools 0.1.1 <pip>
  httpx 0.9.3 <pip>
  humanfriendly 8.2   <pip>
  hyperframe5.2.0 <pip>
  hyperlink 17.3.1<pip>
  idna  2.10  <pip>
  idna  2.7   <pip>
  idna-ssl  1.1.0 <pip>
  immutables0.14  <pip>
  importlib-metadata1.7.0 <pip>
  incremental   17.5.0<pip>
  itsdangerous  1.1.0 <pip>
  jax   0.1.77<pip>
  jaxlib0.1.55<pip>
  jieba 0.42.1<pip>
  Jinja22.11.2<pip>
  jmespath  0.10.0<pip>
  joblib0.16.0<pip>
  jsonpickle1.4.1 <pip>
  jsonschema3.2.0 <pip>
  kafka-python  1.4.7 <pip>
  Keras 2.1.5 <pip>
  Keras-Applications1.0.8 <pip>
  Keras-Preprocessing   1.1.0 <pip>
  kfac  0.2.2 <pip>
  kiwisolver1.2.0 <pip>
  klein 17.10.0   <pip>
  Markdown  3.2.2 <pip>
  MarkupSafe1.1.1 <pip>
  matplotlib3.2.2 <pip>
  mattermostwrapper 2.2   <pip>
  mesh-tensorflow   0.1.16<pip>
  mitie 0.7.0 <pip>
  mpmath1.1.0 <pip>
  msgpack   1.0.0 <pip>
  msgpack-python0.5.4 <pip>
  multidict 4.6.1 <pip>
  networkx  2.4   <pip>
  numpy 1.18.5<pip>
  numpy 1.19.2<pip>
  oauth2client  4.1.3 <pip>
  oauthlib  3.1.0 <pip>
  opencv-python 4.4.0.42  <pip>
  openssl   1.0.2l0https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  opt-einsum3.3.0 <pip>
  packaging 20.4  <pip>
  pandas1.1.2 <pip>
  pi0.1.2 <pip>
  pika  1.1.0 <pip>
  Pillow7.2.0 <pip>
  pip   20.2.3<pip>
  pip   9.0.1py36_1https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  promise   2.3   <pip>
  prompt-toolkit2.0.10<pip>
  protobuf  3.13.0<pip>
  psycopg2-binary   2.8.6 <pip>
  pyasn10.4.8 <pip>
  pyasn1-modules0.2.8 <pip>
  pycparser 2.20  <pip>
  pydot 1.4.1 <pip>
  pygame1.9.6 <pip>
  pyglet1.5.0 <pip>
  PyHamcrest2.0.2 <pip>
  PyJWT 1.7.1 <pip>
  pykwalify 1.7.0 <pip>
  pymongo   3.8.0 <pip>
  pyparsing 2.4.7 <pip>
  pypng 0.0.20<pip>
  pyrsistent0.17.3<pip>
  PySocks   1.7.1 <pip>
  python3.6.2 0https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  python-crfsuite   0.9.7 <pip>
  python-dateutil   2.8.1 <pip>
  python-engineio   3.12.1<pip>
  python-socketio   4.5.1 <pip>
  python-telegram-bot   11.1.0<pip>
  pytz  2019.3<pip>
  PyYAML5.1   <pip>
  pyzmq 17.1.0<pip>
  questionary   1.5.2 <pip>
  rasa  1.10.12   <pip>
  rasa-core 0.9.0 <pip>
  rasa-sdk  1.10.2<pip>
  readline  6.2   2https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  redis 3.5.3 <pip>
  regex 2020.6.8  <pip>
  requests  2.24.0<pip>
  requests-async0.5.0 <pip>
  requests-oauthlib 1.3.0 <pip>
  requests-toolbelt 0.9.1 <pip>
  rfc3986   1.4.0 <pip>
  rocketchat-API0.6.36<pip>
  rsa   4.6   <pip>
  ruamel.yaml   0.16.12   <pip>
  ruamel.yaml.clib  0.2.2 <pip>
  s3transfer0.3.3 <pip>
  sanic 19.12.2   <pip>
  Sanic-Cors0.10.0.post3  <pip>
  sanic-jwt 1.4.1 <pip>
  Sanic-Plugins-Framework   0.9.4 <pip>
  scikit-learn  0.22.2.post1  <pip>
  scipy 1.4.1 <pip>
  setuptools36.4.0   py36_1https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  setuptools50.3.0<pip>
  simplejson3.13.2<pip>
  six   1.15.0<pip>
  sklearn-crfsuite  0.3.6 <pip>
  slackclient   2.9.0 <pip>
  sniffio   1.1.0 <pip>
  SQLAlchemy1.3.19<pip>
  sqlite3.13.00https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  sympy 1.6.2 <pip>
  tabulate  0.8.7 <pip>
  tensor2tensor 1.14.1<pip>
  tensorboard   1.8.0 <pip>
  tensorboard   2.1.1 <pip>
  tensorboard-plugin-wit1.7.0 <pip>
  tensorflow2.1.1 <pip>
  tensorflow1.8.0 <pip>
  tensorflow-addons 0.7.1 <pip>
  tensorflow-cpu1.15.0<pip>
  tensorflow-datasets   3.2.1 <pip>
  tensorflow-estimator  2.1.0 <pip>
  tensorflow-gan2.0.0 <pip>
  tensorflow-hub0.8.0 <pip>
  tensorflow-metadata   0.24.0<pip>
  tensorflow-probability0.7.0 <pip>
  termcolor 1.1.0 <pip>
  terminaltables3.1.0 <pip>
  tf-slim   1.1.0 <pip>
  threadpoolctl 2.1.0 <pip>
  tk8.5.180https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  tqdm  4.45.0<pip>
  twilio6.26.3<pip>
  Twisted   20.3.0<pip>
  typing3.6.2 <pip>
  typing-extensions 3.7.4.3   <pip>
  tzlocal   2.1   <pip>
  ujson 2.0.3 <pip>
  uritemplate   3.0.1 <pip>
  urllib3   1.25.10   <pip>
  urllib3   1.24.3<pip>
  uvloop0.14.0<pip>
  wcwidth   0.2.5 <pip>
  webexteamssdk 1.3   <pip>
  websocket-client  0.54.0<pip>
  websockets8.1   <pip>
  Werkzeug  1.0.1 <pip>
  wheel 0.29.0   py36_0https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  wheel 0.35.1<pip>
  wrapt 1.12.1<pip>
  xz5.2.3 0https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  yarl  1.5.1 <pip>
  zipp  3.1.0 <pip>
  zlib  1.2.110https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  zope.interface5.1.0 <pip>


```


## 二、新建项目

### 2.1 安装rasa，自动生成rasa项目所需文件

```
  # 安装rasa, 由于网络问题，延长超时时间
  # 如果仍然超时异常，多执行几次
  $ mkdir rasa_zh
  $ cd rasa_zh
  $ rasa init --no-prompt # 生成文件
```

> 如果出现 No matching distribution found for tensorflow>=2.1.0异常，可以通过执行pip3 install --upgrade tensorflow rasa命令解决。

### 2.2 测试助手

``` 
  sudo rasa shell
```

## 三、Rasa使用–构建简单聊天机器人

![](img/20200918105542.png)

### 3.1 Rasa工作原理

- 首先，将用户输入的Message传递到Interpreter(Rasa NLU模块)，该模块负责识别Message中的"意图(intent)“和提取所有"实体”(entity)数据；
- 其次，Rasa Core会将Interpreter提取到的意图和识别传给Tracker对象，该对象的主要作用是跟踪会话状态(conversation state)；
- 第三，利用policy记录Tracker对象的当前状态，并选择执行相应的action，其中，这个action是被记录在Track对象中的；
- 最后，将执行action返回的结果输出即完成一次人机交互。

### 3.2 构建NLU样本

#### 3.2.1 data/nlu.md

NLU模型训练样本数据：

```
## intent:greet
- 你好
- 你好啊
- 早上好
- 晚上好
- hello
- hi
- 嗨
- 嗨喽
- 见到你很高兴
- 嘿
- 早
- 上午好
- hello哈喽
- 哈喽哈喽
- hello hello
- 喂喂

## intent:goodbye
- goodbye
- bye
- bye bye
- 88
- 886
- 再见
- 拜
- 拜拜
- 拜拜，下次再聊
- 下次见
- 回头见
- 下次再见
- 下次再聊
- 有空再聊
- 先这样吧
- 好了，就说这么多了
- 好了，先这样
- 没事

## intent:whoareyou
- 你是谁
- 我知道你吗
- 谁
- 我认识你吗
- 这是谁啊
- 是谁
- 请问你是谁
- 请问我认识你吗
- 你是哪位
- 你是？
- 是谁？
- 可以告诉我你的名字吗
- 你叫什么名字

## intent:whattodo
- 你支持什么功能
- 你有什么功能
- 你能干什么
- 你能做什么

## intent:thanks
- 谢谢
- thanks
- thank you
- 真的太感谢你了，帮了我大忙
- 谢谢你帮了我大忙
- 你帮了我大忙，谢谢你小智
- 非常感谢
- 谢了

## intent:deny
- 不
- no
- 不可以
- 不是的
- 不认同
- 否定
- 不是这样子的
- 我不同意你的观点
- 不同意
- 不好
- 你长得很美，就不要想得太美。
- 拒绝
- 不行

## intent:affirm
- 是的
- 当然
- 好的
- ok
- 嗯
- 可以
- 你可以这么做
- 你做得可以啊
- 同意
- 听起来不错
- 是这样的
- 的确是这样子的
- 我同意你的观点
- 对的
- 好滴
- 行
- 还行
- 当然可以

## intent: request_weather
- 天气
- 查询天气
- 帮我查天气信息
- 我想知道[明天](date-time)的天气
- [星期一](date-time)的天气
- [今天](date-time)的天气怎么样
- 帮我查下[后天](date-time)的天气
- 查下[广州](address)的天气怎么样
-  [长沙](address)的天气
- [深圳](address)[明天](date-time)的天气
- 查下[今天](date-time)[上海](address)的天气
- 帮我查查[佛山](address)这[周六](date-time)的天气
```

### 3.3 构建Core样本

#### 3.3.1 data/stories.md

```
## greet
* greet
- utter_answer_greet

## say affirm  with greet
* greet
- utter_answer_greet
* affirm
- utter_answer_affirm

## say affirm 
* affirm
- utter_answer_affirm

## say no with greet
* greet
- utter_answer_greet
* deny
- utter_answer_deny

## say no 
* deny
- utter_answer_deny


## say goodbye
* goodbye
- utter_answer_goodbye

## thanks with greet
* greet
- utter_answer_greet
* thanks
- utter_answer_thanks

## thanks
* thanks
- utter_answer_thanks

## who are you with greet
* greet
- utter_answer_greet
* whoareyou
- utter_answer_whoareyou

## who are you
* whoareyou
- utter_answer_whoareyou

## who are you with greet
* greet
- utter_answer_greet
* whoareyou
- utter_answer_whoareyou

## what to do
* whattodo
- utter_answer_whattodo

## what to do with greet
* greet
- utter_answer_greet
* whattodo
- utter_answer_whattodo

## happy path
* request_weather
- weather_form
- form{"name": "weather_form"}
- form{"name": null}
```

#### 3.3.2 domain.yml

```
intents:
  - affirm
  - deny
  - greet
  - goodbye
  - thanks
  - whoareyou
  - whattodo
  - request_weather

slots:
  date-time:
type: unfeaturized
  address:
type: unfeaturized

entities:
  - date-time
  - address

actions:
  - utter_answer_affirm
  - utter_answer_deny
  - utter_answer_greet
  - utter_answer_goodbye
  - utter_answer_thanks
  - utter_answer_whoareyou
  - utter_answer_whattodo
  - utter_ask_date-time
  - utter_ask_address
  - action_default_fallback

forms:
  - weather_form

responses:
  utter_answer_affirm:
- text: "嗯嗯，好的！"
- text: "嗯嗯，很开心能够帮您解决问题~"
- text: "嗯嗯，还需要什么我能够帮助您的呢？"

  utter_answer_greet:
- text: "您好！请问我可以帮到您吗？"
- text: "您好！很高兴为您服务。请说出您要查询的功能？"

  utter_answer_goodbye:
- text: "再见"
- text: "拜拜"
- text: "虽然我有万般舍不得，但是天下没有不散的宴席~祝您安好！"
- text: "期待下次再见！"
- text: "嗯嗯，下次需要时随时记得我哟~"
- text: "see you!"

  utter_answer_deny:
- text: "主人，您不开心吗？不要离开我哦"
- text: "怎么了，主人？"

  utter_answer_thanks:
- text: "嗯呢。不用客气~"
- text: "这是我应该做的，主人~"
- text: "嗯嗯，合作愉快！"

  utter_answer_whoareyou:
- text: "您好！我是小蒋呀，您的AI智能助理"

  utter_answer_whattodo:
- text: "您好！很高兴为您服务，我目前只支持查询天气哦。"

  utter_ask_date-time:
- text: "请问您要查询哪一天的天气？"

  utter_ask_address:
- text: "请问您要查下哪里的天气？"

  utter_default:
- text: "没听懂，请换种说法吧~"
```

### 3.4 训练NLU和CORE模型

#### 3.4.1 config.yml

训练NLU和Core模型配置文件：

```
language: "zh"

pipeline:
- name: "MitieNLP"
  model: "data/total_word_feature_extractor_zh.dat"
- name: "JiebaTokenizer"
- name: "MitieEntityExtractor"
- name: "EntitySynonymMapper"
- name: "RegexFeaturizer"
- name: "MitieFeaturizer"
- name: "SklearnIntentClassifier"

policies:
  - name: KerasPolicy
    epochs: 500
    max_history: 5
  - name: FallbackPolicy
    fallback_action_name: 'action_default_fallback'
  - name: MemoizationPolicy
    max_history: 5
  - name: FormPolicy
```

#### 3.4.2 模型训练

- 验证

```s
$ python -m rasa data validate
/home/amy/.conda/envs/rasa/lib/python3.6/site-packages/rasa/core/domain.py:151: FutureWarning: No tracker session configuration was found in the loaded domain. Domains without a session config will automatically receive a session expiration time of 60 minutes in Rasa version 2.0 if not configured otherwise.
  session_config = cls._get_session_config(data.get(SESSION_CONFIG_KEY, {}))
2020-09-25 09:15:00 INFO     rasa.validator  - Validating intents...
2020-09-25 09:15:00 INFO     rasa.validator  - Validating uniqueness of intents and stories...
2020-09-25 09:15:00 INFO     rasa.validator  - Validating utterances...
2020-09-25 09:15:00 INFO     rasa.validator  - Story structure validation...
Processed Story Blocks: 100%|█████████████████████████████████████| 17/17 [00:00<00:00, 4121.57it/s, # trackers=1]
2020-09-25 09:15:00 INFO     rasa.core.training.story_conflict  - Considering the preceding 6 turns for conflict analysis.
2020-09-25 09:15:00 INFO     rasa.validator  - No story structure conflicts found.
```

- 运行

当所有样本和配置文件准备好后，接下来就是训练模型了，打开Pycharm命令终端执行下面的命令，该命令会同时训练NLU和Core模型，具体如下：

```
python -m rasa train --config config.yml --domain domain.yml --data data/
```
- 参数介绍

```
usage: rasa train [-h] [-v] [-vv] [--quiet] [--data DATA [DATA ...]]
  [-c CONFIG] [-d DOMAIN] [--out OUT]
  [--augmentation AUGMENTATION] [--DEBUG -plots]
  [--dump-stories] [--fixed-model-name FIXED_MODEL_NAME]
  [--persist-nlu-data] [--force]
  {core,nlu} ...

positional arguments:
  {core,nlu}
core指定训练的模型为core模型
nlu 指定选了的模型为nlu模型

optional arguments:
  -h, --help帮助信息；
  --data指定NLU和Core模型所有样本文件，默认为data目录；
  -c 或--config 指定policy和nlu pipeline配置文件，默认为根目录下config.ym；
  -d 或--domain 指定domain.yml文件，默认为根目录下domain.yml；
  --out 指定模型文件输出路径，默认为自定生成models；
  --augmentation指定训练时需要多少数据augmentation(扩展)，默认为50；
  --DEBUG -plots 一般不用
  --dump-stories是否开启将flattened stories保存到文件，默认为false；
  --fixed-model-name指定生成的模型文件名称，默认none
  --persist-nlu-data是否一定要将nlu训练数据保存到模型，默认为false；
  --force   是否强化模型当训练数据没有变化时，默认为false

Python Logging Options:
  -v, --verbose 开启打印日志；
  -vv, --DEBUG   开启调试模式；
  --quiet   设置日志打印级别为WARNING；
```

## 四、配置Http和Action

### 4.1 credentials.yml

credentials.yml为配置连接到其他服务的详细(认证)信息，当我们需要通过Http的形式访问Rasa Server时，就需要在该文件中配置rest:。rest通道将为您提供一个rest端点（即Rasa Server），用于向其发送消息，响应该请求将发送回bots消息。根据这个文档的说明，当我们请求Rasa Server的URL应为：

http://rasaServerIP:rasaServerPort/webhooks/rest/webhook

该文件内容如下：

```log
# This file contains the credentials for the voice & chat platforms
# which your bot is using.
# https://rasa.com/docs/rasa/user-guide/messaging-and-voice-channels/

rest:
#  # you don't need to provide anything here - this channel doesn't
#  # require any credentials


#facebook:
#  verify: "<verify>"
#  secret: "<your secret>"
#  page-access-token: "<your page access token>"

#slack:
#  slack_token: "<your slack token>"
#  slack_channel: "<the slack channel>"

#socketio:
#  user_message_evt: <event name for user message>
#  bot_message_evt: <event name for but messages>
#  session_persistence: <true/false>

#mattermost:
#  url: "https://<mattermost instance>/api/v4"
#  token: "<bot token>"
#  webhook_url: "<callback URL>"

# This entry is needed if you are using Rasa X. The entry represents credentials 
# for the Rasa X "channel", i.e. Talk to your bot and Share with guest testers.
rasa:
  url: "http://localhost:5002/api"

```

### 4.2 endpoints.yml

如果希望rasa server(注：指rasa core)能够连接到其他web，我们可以再endpoints.yml这个文件中进行配置，比如为了实现custom action，我们就需要在该文件中对action server进行配置，又比如我们将nlu模块放到其他的web项目中，就需要在该文件中配置nlu server等等。endpoints.yml文件内容如下：

```log
# 指定action server的url
# 当然，也可以将action server单独实现在一个web server项目中
# 那么这个url为"https://yourWebIp:yourWebPort/webhook“
action_endpoint:
 url: "http://localhost:5055/webhook"
 
# 配置nlu(单独创建一个web项目):
#  url: "http://10.0.0.153:5000/"

# 配置tracker信息存储服务器
#tracker_store:
#    type: mongod
#    url: mongodb://localhost:27017
#    db: rasa
#    username:
#    password:
#    auth_source: rasa
```


### 4.3 action.py

当Rasa NLU识别到用户输入Message的意图后，Rasa Core对话管理模块就会对其作出回应，而完成这个回应的模块就是action。Rasa Core支持三种action，即default actions、utter actions以及 custom action，这部分我们将在后面详细讲解。另外，本项目为了测试，还对接了图灵机器人和心知天气的API，代码就没有给出了，感兴趣的可以再文末Github源码中查看，这里只给出default action和custom action代码。具体如下：

```python
from typing import Dict, Text, Any, List
from rasa_sdk import Tracker, Action
from rasa_sdk.events import UserUtteranceReverted, Restarted
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction
from requests import (
  ConnectionError,
  HTTPError,
  TooManyRedirects,
  Timeout
)

# action weather_form
class WeatherForm(FormAction):
def name(self) -> Text:
"""Unique identifier of the form"""
return "weather_form"

@staticmethod
def required_slots(tracker: Tracker) -> List[Text]:
"""A list of required slots that the form has to fill"""
return ["date-time", "address"]

def submit(
self,
dispatcher: CollectingDispatcher,
tracker: Tracker,
domain: Dict[Text, Any],
) -> List[Dict]:
"""Define what the form has to do
after all required slots are filled"""
address = tracker.get_slot('address')
date_time = tracker.get_slot('date-time')
print(f"action_default_fallback->address:{address}")
print(f"action_default_fallback->date_time:{date_time}")
dispatcher.utter_message("正在为你查询 {} {}的天气 ing".format(address,date_time))
return [Restarted()]
   
# action_default_fallback
class ActionDefaultFallback(Action):
"""Executes the fallback action and goes back to the previous state
of the dialogue"""
def name(self):
return 'action_default_fallback'

def run(self, dispatcher, tracker, domain):
text = tracker.latest_message.get('text')
print(f"action_default_fallback->text:{text}")
dispatcher.utter_template('utter_default', tracker, silent_fail=True)
return [UserUtteranceReverted()]
```

## 五、启动对话

### 5.1 通过 post 请求 查询 

#### （1）启动Rasa服务

输入下面命令：

```
# 启动rasa服务
# 该服务实现自然语言理解(NLU)和对话管理(Core)功能
# 注：该服务的--port默认为5005，如果使用默认则可以省略

$ python -m rasa run --port 5005 --endpoints endpoints.yml --credentials credentials.yml --DEBUG 
2020-09-21 20:59:49 DEBUG rasa.model  - Extracted model to '/tmp/tmpi1ucn6wd'.
2020-09-21 20:59:49 DEBUG sanic.root  - CORS: Configuring CORS with resources: {'/*': {'origins': [''], 'methods': 'DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT', 'allow_headers': ['.*'], 'expose_headers': 'filename', 'supports_credentials': True, 'max_age': None, 'send_wildcard': False, 'automatic_options': True, 'vary_header': True, 'resources': {'/*': {'origins': ''}}, 'intercept_exceptions': True, 'always_send': True}}
2020-09-21 20:59:49 DEBUG rasa.core.utils  - Available web server routes: 
/webhooks/rasa GETcustom_webhook_RasaChatInput.health
/webhooks/rasa/webhook POST   custom_webhook_RasaChatInput.receive
/webhooks/rest GETcustom_webhook_RestInput.health
/webhooks/rest/webhook POST   custom_webhook_RestInput.receive
/  GEThello
2020-09-21 20:59:49 INFO root  - Starting Rasa server on http://localhost:5005
2020-09-21 20:59:49 DEBUG rasa.core.utils  - Using the default number of Sanic workers (1).
2020-09-21 20:59:49 INFO root  - Enabling coroutine DEBUG ging. Loop id 77433480.
2020-09-21 20:59:49 DEBUG rasa.model  - Extracted model to '/tmp/tmprhshce_e'.
2020-09-21 20:59:52 INFO rasa.nlu.components  - Added 'MitieNLP' to component cache. Key 'MitieNLP-/web/workspace/yangkm/python_wp/nlu/DSwP/rasa_zh_my/data/total_word_feature_extractor_zh.dat'.
2020-09-21 20:59:52 DEBUG rasa.core.tracker_store  - Connected to InMemoryTrackerStore.
2020-09-21 20:59:52 DEBUG rasa.core.lock_store  - Connected to lock store 'InMemoryLockStore'.
2020-09-21 20:59:52 DEBUG rasa.model  - Extracted model to '/tmp/tmpxex779pr'.
2020-09-21 20:59:52 DEBUG pykwalify.compat  - Using yaml library: /home/amy/.conda/envs/rasa/lib/python3.6/site-packages/ruamel/yaml/__init__.py
/home/amy/.conda/envs/rasa/lib/python3.6/site-packages/rasa/core/policies/keras_policy.py:265: FutureWarning: 'KerasPolicy' is deprecated and will be removed in version 2.0. Use 'TEDPolicy' instead.
  current_epoch=meta["epochs"],
2020-09-21 20:59:55 INFO rasa.core.policies.ensemble  - MappingPolicy not included in policy ensemble. Default intents 'restart and back will not trigger actions 'action_restart' and 'action_back'.
2020-09-21 20:59:55 DEBUG rasa.core.nlg.generator  - Instantiated NLG to 'TemplatedNaturalLanguageGenerator'.
2020-09-21 20:59:55 INFO root  - Rasa server is up and running.

```

- 参数说明

```shell
usage: rasa run [-h] [-v] [-vv] [--quiet] [-m MODEL] [--log-file LOG_FILE]
[--endpoints ENDPOINTS] [-p PORT] [-t AUTH_TOKEN]
[--cors [CORS [CORS ...]]] [--enable-api]
[--remote-storage REMOTE_STORAGE]
[--ssl-certificate SSL_CERTIFICATE]
[--ssl-keyfile SSL_KEYFILE] [--ssl-ca-file SSL_CA_FILE]
[--ssl-password SSL_PASSWORD] [--credentials CREDENTIALS]
[--connector CONNECTOR] [--jwt-secret JWT_SECRET]
[--jwt-method JWT_METHOD]
{actions} ... [model-as-positional-argument]

positional arguments:
  {actions}
actions 运行action server
  model-as-positional-argument 

optional arguments:
  -h, --help  
  						显示帮助信息；
  -m MODEL, --model MODEL 
  						指定训练好的模型路径，默认使用models目录。
  						如果模型存储路径改变，则需要该参数指定；
  --log-file LOG_FILE 
  						指定保存logs文件，默认为None；
  --endpoints ENDPOINTS   
  						指定endpoints.yml文件路径，默认为None;

Python Logging Options:
  -v, --verbose 
  						设置日志等级为INFO;
  -vv, --DEBUG   
  						开启调试
  --quiet   
  						设置日志等级为WARNING，默认为None;

Server Settings:
  -p PORT, --port PORT  
  						设置运行rasa serve的端口号，默认为5005；
  -t AUTH_TOKEN, --auth-token AUTH_TOKEN
开启token身份验证，默认为None;
  --cors [CORS [CORS ...]]
  --enable-api  
  						启动web服务器API通道，默认值:False；
  --remote-storage REMOTE_STORAGE
设置rasa模型远程位置，如果有的话；
  --ssl-certificate SSL_CERTIFICATE
设置SSL证书，默认为None;
  --ssl-keyfile SSL_KEYFILE
设置SSL密钥文件，默认为None;
  --ssl-ca-file SSL_CA_FILE
设置CA文件便于SSL证书验证，默认为None;
  --ssl-password SSL_PASSWORD
  						设置SSL密钥文件密码，默认为None;

Channels:
  --credentials CREDENTIALS
指定credentials.yml文件路径；
  --connector CONNECTOR
Service to connect to. (default: None)

JWT Authentication:
  --jwt-secret JWT_SECRET
指定jwt-secret；
  --jwt-method JWT_METHOD
指定jwt-method；
```

#### （2）启动Action服务

```
# 启动action服务
# 注：该服务的--port默认为5055，如果使用默认则可以省略
$ python -m rasa run actions --port 5055 --actions actions --DEBUG 
2020-09-21 21:39:28 INFO rasa_sdk.endpoint  - Starting action endpoint server...
2020-09-21 21:39:28 INFO rasa_sdk.executor  - Registered function for 'action_default_fallback'.
2020-09-21 21:39:28 INFO rasa_sdk.executor  - Registered function for 'weather_form'.
2020-09-21 21:39:28 INFO rasa_sdk.endpoint  - Action endpoint is up and running on http://localhost:5055
2020-09-21 21:39:28 DEBUG rasa_sdk.utils  - Using the default number of Sanic workers (1).

```
- 参数说明

```
usage: rasa run actions [-h] [-v] [-vv] [--quiet] [-p PORT]
[--cors [CORS [CORS ...]]] [--actions ACTIONS]
[--ssl-keyfile SSL_KEYFILE]
[--ssl-certificate SSL_CERTIFICATE]
[--ssl-password SSL_PASSWORD]

optional arguments:
  -h, --help
  						显示帮助信息
  -p PORT, --port PORT  
  						指定action server的端口号，默认为5055；
  --cors [CORS [CORS ...]]
开启CORS;
  --actions ACTIONS 
  						指定action.py等文件所在包路径；
  --ssl-certificate SSL_CERTIFICATE
设置SSL证书，默认为None;
  --ssl-keyfile SSL_KEYFILE
设置SSL密钥文件，默认为None;
  --ssl-ca-file SSL_CA_FILE
设置CA文件便于SSL证书验证，默认为None;

Python Logging Options:
  -v, --verbose 
  						设置日志等级为INFO;
  -vv, --DEBUG   
  						开启调试
  --quiet   
  						设置日志等级为WARNING，默认为None;
```

#### （3） 效果演示

- 另外开启一个 窗口，输入：

```
$ curl -X POST localhost:5005/webhooks/rest/webhook -d '{"message":"询深圳周五的天气","sender":"1"}' | python -m json.tool
  % Total% Received % Xferd  Average Speed   TimeTime Time  Current
 Dload  Upload   Total   SpentLeft  Speed
100   155  100   104  10051   2355   1155 --:--:-- --:--:-- --:--:--  2363
[
{
"recipient_id": "1",
"text": "\u6682\u4e0d\u652f\u6301\u67e5\u8be2 ['\u6df1\u5733'] \u7684\u5929\u6c14"
}
]

```

- Rasa Server 反映

```
2020-09-21 21:40:06 DEBUG rasa.core.lock_store  - Issuing ticket for conversation '1'.
2020-09-21 21:40:06 DEBUG rasa.core.lock_store  - Acquiring lock for conversation '1'.
2020-09-21 21:40:06 DEBUG rasa.core.lock_store  - Acquired lock for conversation '1'.
2020-09-21 21:40:06 DEBUG rasa.core.tracker_store  - Recreating tracker for id '1'
2020-09-21 21:40:06 DEBUG rasa.core.processor  - Received user message '询深圳周五的天气' with intent '{'name': 'request_weather', 'confidence': 0.49303004566904013}' and entities '[{'entity': 'address', 'value': '深圳', 'start': 1, 'end': 3, 'confidence': None, 'extractor': 'MitieEntityExtractor'}, {'entity': 'date-time', 'value': '周五', 'start': 3, 'end': 5, 'confidence': None, 'extractor': 'MitieEntityExtractor'}]'
2020-09-21 21:40:06 DEBUG rasa.core.processor  - Current slot values: 
	address: 深圳
	date-time: 周五
	requested_slot: None
2020-09-21 21:40:06 DEBUG rasa.core.processor  - Logged UserUtterance - tracker now has 38 events.
2020-09-21 21:40:06 DEBUG rasa.core.policies.fallback  - NLU confidence threshold met, confidence of fallback action set to core threshold (0.3).
2020-09-21 21:40:06 DEBUG rasa.core.policies.memoization  - Current tracker state [None, None, None, {}, {'prev_action_listen': 1.0, 'entity_date-time': 1.0, 'entity_address': 1.0, 'intent_request_weather': 1.0}]
2020-09-21 21:40:06 DEBUG rasa.core.policies.memoization  - There is no memorised next action
2020-09-21 21:40:06 DEBUG rasa.core.policies.form_policy  - There is no active form
2020-09-21 21:40:06 DEBUG rasa.core.policies.ensemble  - Predicted next action using policy_0_KerasPolicy
2020-09-21 21:40:06 DEBUG rasa.core.processor  - Predicted next action 'weather_form' with confidence 0.99.
2020-09-21 21:40:06 DEBUG rasa.core.actions.action  - Calling action endpoint to run action 'weather_form'.
2020-09-21 21:40:06 DEBUG rasa.core.processor  - Action 'weather_form' ended with events '[BotUttered('暂不支持查询 ['深圳'] 的天气', {"elements": null, "quick_replies": null, "buttons": null, "attachment": null, "image": null, "custom": null}, {}, 1600695606.1912577), <rasa.core.events.Form object at 0x7f46445940b8>, <rasa.core.events.SlotSet object at 0x7f4644594278>, <rasa.core.events.SlotSet object at 0x7f4644594240>, <rasa.core.events.SlotSet object at 0x7f4644594160>, <rasa.core.events.SlotSet object at 0x7f46445941d0>, <rasa.core.events.Restarted object at 0x7f46445944a8>, <rasa.core.events.Form object at 0x7f4644594128>, <rasa.core.events.SlotSet object at 0x7f46445944e0>]'.
2020-09-21 21:40:06 DEBUG rasa.core.processor  - Current slot values: 
	address: None
	date-time: None
	requested_slot: None
2020-09-21 21:40:06 DEBUG rasa.core.processor  - Predicted next action 'action_listen' with confidence 1.00.
2020-09-21 21:40:06 DEBUG rasa.core.processor  - Action 'action_listen' ended with events '[]'.
2020-09-21 21:40:06 DEBUG rasa.core.lock_store  - Deleted lock for conversation '1'.

```

- Action Server 反应




### 5.2 聊天窗 式 对话

#### （一）启动Action服务

```s
  # 启动action服务
  # 注：该服务的--port默认为5055，如果使用默认则可以省略
  $ python -m rasa run actions --port 5055 --actions actions --debug
  2020-09-21 21:39:28 INFO rasa_sdk.endpoint  - Starting action endpoint server...
  2020-09-21 21:39:28 INFO rasa_sdk.executor  - Registered function for 'action_default_fallback'.
  2020-09-21 21:39:28 INFO rasa_sdk.executor  - Registered function for 'weather_form'.
  2020-09-21 21:39:28 INFO rasa_sdk.endpoint  - Action endpoint is up and running on http://localhost:5055
  2020-09-21 21:39:28 DEBUG rasa_sdk.utils  - Using the default number of Sanic workers (1).
```
- 参数说明

```s
  usage: rasa run actions [-h] [-v] [-vv] [--quiet] [-p PORT]
  [--cors [CORS [CORS ...]]] [--actions ACTIONS]
  [--ssl-keyfile SSL_KEYFILE]
  [--ssl-certificate SSL_CERTIFICATE]
  [--ssl-password SSL_PASSWORD]

  optional arguments:
    -h, --help
                显示帮助信息
    -p PORT, --port PORT  
                指定action server的端口号，默认为5055；
    --cors [CORS [CORS ...]]
  开启CORS;
    --actions ACTIONS 
                指定action.py等文件所在包路径；
    --ssl-certificate SSL_CERTIFICATE
  设置SSL证书，默认为None;
    --ssl-keyfile SSL_KEYFILE
  设置SSL密钥文件，默认为None;
    --ssl-ca-file SSL_CA_FILE
  设置CA文件便于SSL证书验证，默认为None;

  Python Logging Options:
    -v, --verbose 
                设置日志等级为INFO;
    -vv, --DEBUG   
                开启调试
    --quiet   
                设置日志等级为WARNING，默认为None;
```


#### （二） 开启对话窗口

```python
  $ rasa shell
  2020-09-22 10:05:42 INFO root  - Connecting to channel 'cmdline' which was specified by the '--connector' argument. Any other channels will be ignored. To connect to all given channels, omit the '--connector' argument.
  2020-09-22 10:05:42 INFO root  - Starting Rasa server on http://localhost:5005
  2020-09-22 10:05:46 INFO rasa.nlu.components  - Added 'MitieNLP' to component cache. Key 'MitieNLP-/web/workspace/yangkm/python_wp/nlu/DSwP/rasa_zh_my/data/total_word_feature_extractor_zh.dat'.
  /home/amy/.conda/envs/rasa/lib/python3.6/site-packages/rasa/core/policies/keras_policy.py:265: FutureWarning: 'KerasPolicy' is deprecated and will be removed in version 2.0. Use 'TEDPolicy' instead.
current_epoch=meta["epochs"],
  2020-09-22 10:05:50 INFO rasa.core.policies.ensemble  - MappingPolicy not included in policy ensemble. Default intents 'restart and back will not trigger actions 'action_restart' and 'action_back'.
  2020-09-22 10:05:50 INFO root  - Rasa server is up and running.
  Bot loaded. Type a message and press enter (use '/stop' to exit): 
  Your input ->  
```

#### （三） 

1. 场景一

> 对话内容
```s
  Your input ->  查询天气  
  请问您要查询哪一天的天气？
  Your input ->  大后天 
  请问您要查下哪里的天气？
  Your input ->  哈尔滨 
  正在为你查询 哈尔滨 大后天的天气 ing
```

> 后台结果
```log
2020-09-22 10:06:45 DEBUG rasa_sdk.forms  - Validating user input '{'intent': {'name': 'request_weather', 'confidence': 0.6752573663579792}, 'entities': [], 'intent_ranking': [{'name': 'request_weather', 'confidence': 0.6752573663579792}, {'name': 'deny', 'confidence': 0.08473735571280376}, {'name': 'affirm', 'confidence': 0.07509502417330365}, {'name': 'greet', 'confidence': 0.05720770660563366}, {'name': 'thanks', 'confidence': 0.03785058670945419}, {'name': 'goodbye', 'confidence': 0.03603811968623011}, {'name': 'whattodo', 'confidence': 0.02145608716337933}, {'name': 'whoareyou', 'confidence': 0.01235775359121649}], 'text': '查询天气'}'
2020-09-22 10:06:45 DEBUG rasa_sdk.forms  - Validating extracted slots: {}
2020-09-22 10:06:45 DEBUG rasa_sdk.forms  - Request next slot 'date-time'
2020-09-22 10:06:45 DEBUG rasa_sdk.executor  - Finished running 'weather_form'
2020-09-22 10:06:51 DEBUG rasa_sdk.executor  - Received request to run 'weather_form'
2020-09-22 10:06:51 DEBUG rasa_sdk.forms  - The form '{'name': 'weather_form', 'validate': True, 'rejected': False, 'trigger_message': {'intent': {'name': 'request_weather', 'confidence': 0.6752573663579792}, 'entities': [], 'intent_ranking': [{'name': 'request_weather', 'confidence': 0.6752573663579792}, {'name': 'deny', 'confidence': 0.08473735571280376}, {'name': 'affirm', 'confidence': 0.07509502417330365}, {'name': 'greet', 'confidence': 0.05720770660563366}, {'name': 'thanks', 'confidence': 0.03785058670945419}, {'name': 'goodbye', 'confidence': 0.03603811968623011}, {'name': 'whattodo', 'confidence': 0.02145608716337933}, {'name': 'whoareyou', 'confidence': 0.01235775359121649}], 'text': '查询天气'}}' is active
2020-09-22 10:06:51 DEBUG rasa_sdk.forms  - Validating user input '{'intent': {'name': 'request_weather', 'confidence': 0.9364390143513037}, 'entities': [{'entity': 'date-time', 'value': '大后天', 'start': 0, 'end': 3, 'confidence': None, 'extractor': 'MitieEntityExtractor'}], 'intent_ranking': [{'name': 'request_weather', 'confidence': 0.9364390143513037}, {'name': 'goodbye', 'confidence': 0.02844964603391187}, {'name': 'greet', 'confidence': 0.01407263502906878}, {'name': 'affirm', 'confidence': 0.0070474768555266265}, {'name': 'thanks', 'confidence': 0.006487673729264043}, {'name': 'whattodo', 'confidence': 0.0032935182462639516}, {'name': 'deny', 'confidence': 0.0027266823314896858}, {'name': 'whoareyou', 'confidence': 0.0014833534231716052}], 'text': '大后天'}'
2020-09-22 10:06:51 DEBUG rasa_sdk.forms  - Trying to extract requested slot 'date-time' ...
2020-09-22 10:06:51 DEBUG rasa_sdk.forms  - Got mapping '{'type': 'from_entity', 'entity': 'date-time', 'intent': [], 'not_intent': [], 'role': None, 'group': None}'
2020-09-22 10:06:51 DEBUG rasa_sdk.forms  - Successfully extracted '大后天' for requested slot 'date-time'
2020-09-22 10:06:51 DEBUG rasa_sdk.forms  - Validating extracted slots: {'date-time': '大后天'}
2020-09-22 10:06:51 DEBUG rasa_sdk.forms  - Request next slot 'address'
2020-09-22 10:06:51 DEBUG rasa_sdk.executor  - Finished running 'weather_form'
2020-09-22 10:07:13 DEBUG rasa_sdk.executor  - Received request to run 'weather_form'
2020-09-22 10:07:13 DEBUG rasa_sdk.forms  - The form '{'name': 'weather_form', 'validate': True, 'rejected': False, 'trigger_message': {'intent': {'name': 'request_weather', 'confidence': 0.6752573663579792}, 'entities': [], 'intent_ranking': [{'name': 'request_weather', 'confidence': 0.6752573663579792}, {'name': 'deny', 'confidence': 0.08473735571280376}, {'name': 'affirm', 'confidence': 0.07509502417330365}, {'name': 'greet', 'confidence': 0.05720770660563366}, {'name': 'thanks', 'confidence': 0.03785058670945419}, {'name': 'goodbye', 'confidence': 0.03603811968623011}, {'name': 'whattodo', 'confidence': 0.02145608716337933}, {'name': 'whoareyou', 'confidence': 0.01235775359121649}], 'text': '查询天气'}}' is active
2020-09-22 10:07:13 DEBUG rasa_sdk.forms  - Validating user input '{'intent': {'name': 'request_weather', 'confidence': 0.2875610430157071}, 'entities': [{'entity': 'address', 'value': '哈尔滨', 'start': 0, 'end': 3, 'confidence': None, 'extractor': 'MitieEntityExtractor'}], 'intent_ranking': [{'name': 'request_weather', 'confidence': 0.2875610430157071}, {'name': 'greet', 'confidence': 0.2872763354473037}, {'name': 'goodbye', 'confidence': 0.12110821662968022}, {'name': 'affirm', 'confidence': 0.0787910767204003}, {'name': 'thanks', 'confidence': 0.0736496158800513}, {'name': 'whattodo', 'confidence': 0.06899456035142955}, {'name': 'deny', 'confidence': 0.05765607701548124}, {'name': 'whoareyou', 'confidence': 0.024963074939946338}], 'text': '哈尔滨'}'
2020-09-22 10:07:13 DEBUG rasa_sdk.forms  - Trying to extract requested slot 'address' ...
2020-09-22 10:07:13 DEBUG rasa_sdk.forms  - Got mapping '{'type': 'from_entity', 'entity': 'address', 'intent': [], 'not_intent': [], 'role': None, 'group': None}'
2020-09-22 10:07:13 DEBUG rasa_sdk.forms  - Successfully extracted '哈尔滨' for requested slot 'address'
2020-09-22 10:07:13 DEBUG rasa_sdk.forms  - Validating extracted slots: {'address': '哈尔滨'}
2020-09-22 10:07:13 DEBUG rasa_sdk.forms  - No slots left to request, all required slots are filled:
	date-time: 大后天
	address: 哈尔滨
2020-09-22 10:07:13 DEBUG rasa_sdk.forms  - Submitting the form 'weather_form'
action_default_fallback->address:哈尔滨
action_default_fallback->date_time:大后天
2020-09-22 10:07:13 DEBUG rasa_sdk.forms  - Deactivating the form 'weather_form'
2020-09-22 10:07:13 DEBUG rasa_sdk.executor  - Finished running 'weather_form'
```

2. 场景二

> 对话内容
```s
  Your input ->  明天的天气 
  请问您要查下哪里的天气？
  Your input ->  深圳   
  正在为你查询 深圳 明天的天气 ing
  Your input ->  深圳的天气 
  请问您要查询哪一天的天气？
  Your input ->  后台   
  正在为你查询 深圳 后台的天气 ing
```
> 后台结果
```log
2020-09-22 10:07:23 DEBUG     rasa_sdk.forms  - Activated the form 'weather_form'
2020-09-22 10:07:23 DEBUG     rasa_sdk.forms  - Validating pre-filled required slots: {'date-time': '明天'}
2020-09-22 10:07:23 DEBUG     rasa_sdk.forms  - Validating user input '{'intent': {'name': 'request_weather', 'confidence': 0.859133506308341}, 'entities': [{'entity': 'date-time', 'value': '明天', 'start': 0, 'end': 2, 'confidence': None, 'extractor': 'MitieEntityExtractor'}], 'intent_ranking': [{'name': 'request_weather', 'confidence': 0.859133506308341}, {'name': 'deny', 'confidence': 0.03325788518527375}, {'name': 'affirm', 'confidence': 0.03310349827249277}, {'name': 'greet', 'confidence': 0.02768090617777744}, {'name': 'thanks', 'confidence': 0.018412903569335396}, {'name': 'goodbye', 'confidence': 0.016649633914462992}, {'name': 'whattodo', 'confidence': 0.005943813262005036}, {'name': 'whoareyou', 'confidence': 0.005817853310311661}], 'text': '明天的天气'}'
2020-09-22 10:07:23 DEBUG     rasa_sdk.forms  - Extracted '明天' for extra slot 'date-time'.
2020-09-22 10:07:23 DEBUG     rasa_sdk.forms  - Validating extracted slots: {'date-time': '明天'}
2020-09-22 10:07:23 DEBUG     rasa_sdk.forms  - Request next slot 'address'
2020-09-22 10:07:23 DEBUG     rasa_sdk.executor  - Finished running 'weather_form'
2020-09-22 10:07:26 DEBUG     rasa_sdk.executor  - Received request to run 'weather_form'
2020-09-22 10:07:26 DEBUG     rasa_sdk.forms  - The form '{'name': 'weather_form', 'validate': True, 'rejected': False, 'trigger_message': {'intent': {'name': 'request_weather', 'confidence': 0.859133506308341}, 'entities': [{'entity': 'date-time', 'value': '明天', 'start': 0, 'end': 2, 'confidence': None, 'extractor': 'MitieEntityExtractor'}], 'intent_ranking': [{'name': 'request_weather', 'confidence': 0.859133506308341}, {'name': 'deny', 'confidence': 0.03325788518527375}, {'name': 'affirm', 'confidence': 0.03310349827249277}, {'name': 'greet', 'confidence': 0.02768090617777744}, {'name': 'thanks', 'confidence': 0.018412903569335396}, {'name': 'goodbye', 'confidence': 0.016649633914462992}, {'name': 'whattodo', 'confidence': 0.005943813262005036}, {'name': 'whoareyou', 'confidence': 0.005817853310311661}], 'text': '明天的天气'}}' is active
2020-09-22 10:07:26 DEBUG     rasa_sdk.forms  - Validating user input '{'intent': {'name': 'request_weather', 'confidence': 0.5679284847462811}, 'entities': [{'entity': 'address', 'value': '深圳', 'start': 0, 'end': 2, 'confidence': None, 'extractor': 'MitieEntityExtractor'}], 'intent_ranking': [{'name': 'request_weather', 'confidence': 0.5679284847462811}, {'name': 'greet', 'confidence': 0.1836000885202259}, {'name': 'thanks', 'confidence': 0.06674624179701746}, {'name': 'deny', 'confidence': 0.050381853263852465}, {'name': 'affirm', 'confidence': 0.044965754284358395}, {'name': 'goodbye', 'confidence': 0.04295301432924679}, {'name': 'whattodo', 'confidence': 0.03497770707600284}, {'name': 'whoareyou', 'confidence': 0.00844685598301393}], 'text': '深圳'}'
2020-09-22 10:07:26 DEBUG     rasa_sdk.forms  - Trying to extract requested slot 'address' ...
2020-09-22 10:07:26 DEBUG     rasa_sdk.forms  - Got mapping '{'type': 'from_entity', 'entity': 'address', 'intent': [], 'not_intent': [], 'role': None, 'group': None}'
2020-09-22 10:07:26 DEBUG     rasa_sdk.forms  - Successfully extracted '深圳' for requested slot 'address'
2020-09-22 10:07:26 DEBUG     rasa_sdk.forms  - Validating extracted slots: {'address': '深圳'}
2020-09-22 10:07:26 DEBUG     rasa_sdk.forms  - No slots left to request, all required slots are filled:
	date-time: 明天
	address: 深圳
2020-09-22 10:07:26 DEBUG     rasa_sdk.forms  - Submitting the form 'weather_form'
action_default_fallback->address:深圳
action_default_fallback->date_time:明天
2020-09-22 10:07:26 DEBUG     rasa_sdk.forms  - Deactivating the form 'weather_form'
2020-09-22 10:07:26 DEBUG     rasa_sdk.executor  - Finished running 'weather_form'
2020-09-22 10:07:33 DEBUG     rasa_sdk.executor  - Received request to run 'weather_form'
2020-09-22 10:07:33 DEBUG     rasa_sdk.forms  - There is no active form
2020-09-22 10:07:33 DEBUG     rasa_sdk.forms  - Activated the form 'weather_form'
2020-09-22 10:07:33 DEBUG     rasa_sdk.forms  - Validating pre-filled required slots: {'address': '深圳'}
2020-09-22 10:07:33 DEBUG     rasa_sdk.forms  - Validating user input '{'intent': {'name': 'request_weather', 'confidence': 0.6615162196934451}, 'entities': [{'entity': 'address', 'value': '深圳', 'start': 0, 'end': 2, 'confidence': None, 'extractor': 'MitieEntityExtractor'}], 'intent_ranking': [{'name': 'request_weather', 'confidence': 0.6615162196934451}, {'name': 'affirm', 'confidence': 0.0900704675472}, {'name': 'deny', 'confidence': 0.08134485744568894}, {'name': 'greet', 'confidence': 0.06553939061503906}, {'name': 'thanks', 'confidence': 0.046344364587568965}, {'name': 'goodbye', 'confidence': 0.0333352458887086}, {'name': 'whattodo', 'confidence': 0.014025037948307274}, {'name': 'whoareyou', 'confidence': 0.007824416274041972}], 'text': '深圳的天气'}'
2020-09-22 10:07:33 DEBUG     rasa_sdk.forms  - Extracted '深圳' for extra slot 'address'.
2020-09-22 10:07:33 DEBUG     rasa_sdk.forms  - Validating extracted slots: {'address': '深圳'}
2020-09-22 10:07:33 DEBUG     rasa_sdk.forms  - Request next slot 'date-time'
2020-09-22 10:07:33 DEBUG     rasa_sdk.executor  - Finished running 'weather_form'

```
