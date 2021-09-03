# 【关于 rasa中文对话系统】那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 
> 学习项目：https://github.com/zqhZY/_rasa_chatbot

## 目录

- [【关于 rasa中文对话系统】那些你不知道的事](#关于-rasa中文对话系统那些你不知道的事)
  - [目录](#目录)
  - [安装 Rasa 内容](#安装-rasa-内容)
    - [安装 中文版本 的 Rasa-nlu](#安装-中文版本-的-rasa-nlu)
      - [安装过程中遇到的问题](#安装过程中遇到的问题)
    - [sklearn  和 MITIE 库 安装](#sklearn--和-mitie-库-安装)
      - [安装过程中遇到的问题](#安装过程中遇到的问题-1)
    - [安装 rasa_core](#安装-rasa_core)
  - [项目初尝试](#项目初尝试)
    - [获取项目](#获取项目)
    - [项目目录介绍](#项目目录介绍)
    - [模型训练](#模型训练)
    - [测试 rasa nlu](#测试-rasa-nlu)
    - [训练对话](#训练对话)
    - [在线模式下的对话训练](#在线模式下的对话训练)
    - [测试 对话 功能](#测试-对话-功能)
  - [项目核心内容分析](#项目核心内容分析)
    - [ivr_chatbot.yml 配置文件 分析](#ivr_chatbotyml-配置文件-分析)
    - [./data/mobile_nlu_data.json 意图识别和实体识别 训练数据 分析](#datamobile_nlu_datajson-意图识别和实体识别-训练数据-分析)
    - [mobile_story.md 对话设置 文件 分析](#mobile_storymd-对话设置-文件-分析)
    - [mobile_domain.yml 定义域 文件 分析](#mobile_domainyml-定义域-文件-分析)
  - [参考资料](#参考资料)


## 安装 Rasa 内容

> 温馨提示：由于 安装 Rasa 过程中，会安装各种 乱七八糟的 依赖库（eg：tensorflow 2.0，...），导致 安装失败，所以建议 用 conda ，新建 一个 conda 环境，然后在 该环境上面开发。

### 安装 中文版本 的 Rasa-nlu 

```
  $ git clone https://github.com/crownpku/Rasa_NLU_Chi.git
  $ cd rasa_nlu
  $ pip install -r requirements.txt
  $ python setup.py install
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

#### 安装过程中遇到的问题

- 问题一：win10下安装MITIE

> 依赖包：MITIE安装需要有Visual Studio环境、cmake、boost。注意，这三种缺一不可。

1. 安装Visual Studio

做过C#开发的童鞋，肯定很熟悉Visual Studio，即VS。windows 的集成开发环境。安装该环境的同时，它会附带安装很全的windows的类库。后面boost库运行的时候，需要使用其中的类库。

具体安装过程很简单，完全傻瓜式安装即可，下一步下一步搞定。这里提供一个下载地址：

http://download.microsoft.com/download/0/7/5/0755898A-ED1B-4E11-BC04-6B9B7D82B1E4/VS2013_RTM_ULT_CHS.iso

2. 安装cmake

官网下载：https://cmake.org/download/

解压后把bin目录路径，配置到path环境变量中。
例如：D:\develop-environment\cmake-3.12.3-win64-x64\bin
执行文件为：

```
	cmake.exe
	cmake-gui.ext
	cmcldeps.exe
	cpack.exe
```

3. 安装 boost

官网下载：https://www.boost.org/

因为官网下载需要翻墙，百度网盘提供一个： https://pan.baidu.com/s/1LOgKv_S-JdvUNZ2UQBNCjA 提取码: eeuw

我本机boost的解压目录为：
D:\develop-environment\boost\boost_1_67_0

```
	cd D:\develop-environment\boost\boost_1_67_0\tools\build
	bootstrap.bat
	.\b2 --prefix=D:\develop-environment\boost\bin install
```

- 问题二： Centos7 下安装MITIE

> 出现问题
```s
pip install mitie
Collecting mitie
  Using cached https://files.pythonhosted.org/packages/80/e9/4481c5e6233b8b93acccaacf595bc8e11f40d6ac2e6f6e70b7a62693f9ea/mitie-0.7.36.tar.gz
Building wheels for collected packages: mitie
  Building wheel for mitie (setup.py) ... error
  ERROR: Command errored out with exit status 1:
   command: /root/anaconda3/envs/acsz/bin/python -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-zcc2vtwz/mitie/setup.py'"'"'; __file__='"'"'/tmp/pip-install-zcc2vtwz/mitie/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' bdist_wheel -d /tmp/pip-wheel-j9l0ta79 --python-tag cp37
       cwd: /tmp/pip-install-zcc2vtwz/mitie/
  Complete output (30 lines):
  running bdist_wheel
  running build
  make -C mitielib
  make[1]: Entering directory `/tmp/pip-install-zcc2vtwz/mitie/mitielib'
  Compiling src/mitie.cpp
  make[1]: g++: Command not found
  make[1]: *** [src/mitie.o] Error 127
  make[1]: Leaving directory `/tmp/pip-install-zcc2vtwz/mitie/mitielib'
  make: *** [mitielib] Error 2
  Traceback (most recent call last):
    File "<string>", line 1, in <module>
    File "/tmp/pip-install-zcc2vtwz/mitie/setup.py", line 67, in <module>
      'Programming Language :: Python :: 3.5',
    File "/root/anaconda3/envs/acsz/lib/python3.7/distutils/core.py", line 148, in setup
      dist.run_commands()
    File "/root/anaconda3/envs/acsz/lib/python3.7/distutils/dist.py", line 966, in run_commands
      self.run_command(cmd)
    File "/root/anaconda3/envs/acsz/lib/python3.7/distutils/dist.py", line 985, in run_command
      cmd_obj.run()
    File "/root/anaconda3/envs/acsz/lib/python3.7/site-packages/wheel/bdist_wheel.py", line 192, in run
      self.run_command('build')
    File "/root/anaconda3/envs/acsz/lib/python3.7/distutils/cmd.py", line 313, in run_command
      self.distribution.run_command(command)
    File "/root/anaconda3/envs/acsz/lib/python3.7/distutils/dist.py", line 985, in run_command
      cmd_obj.run()
    File "/tmp/pip-install-zcc2vtwz/mitie/setup.py", line 36, in run
      subprocess.check_call(['make', 'mitielib'])
    File "/root/anaconda3/envs/acsz/lib/python3.7/subprocess.py", line 347, in check_call
      raise CalledProcessError(retcode, cmd)
  subprocess.CalledProcessError: Command '['make', 'mitielib']' returned non-zero exit status 2.
  ----------------------------------------
  ERROR: Failed building wheel for mitie
  Running setup.py clean for mitie
Failed to build mitie
Installing collected packages: mitie
  Running setup.py install for mitie ... error
    ERROR: Command errored out with exit status 1:
     command: /root/anaconda3/envs/acsz/bin/python -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-zcc2vtwz/mitie/setup.py'"'"'; __file__='"'"'/tmp/pip-install-zcc2vtwz/mitie/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' install --record /tmp/pip-record-pjnzlmaq/install-record.txt --single-version-externally-managed --compile
         cwd: /tmp/pip-install-zcc2vtwz/mitie/
    Complete output (32 lines):
    running install
    running build
    make -C mitielib
    make[1]: Entering directory `/tmp/pip-install-zcc2vtwz/mitie/mitielib'
    Compiling src/mitie.cpp
    make[1]: g++: Command not found
    make[1]: *** [src/mitie.o] Error 127
    make[1]: Leaving directory `/tmp/pip-install-zcc2vtwz/mitie/mitielib'
    make: *** [mitielib] Error 2
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/tmp/pip-install-zcc2vtwz/mitie/setup.py", line 67, in <module>
        'Programming Language :: Python :: 3.5',
      File "/root/anaconda3/envs/acsz/lib/python3.7/distutils/core.py", line 148, in setup
        dist.run_commands()
      File "/root/anaconda3/envs/acsz/lib/python3.7/distutils/dist.py", line 966, in run_commands
        self.run_command(cmd)
      File "/root/anaconda3/envs/acsz/lib/python3.7/distutils/dist.py", line 985, in run_command
        cmd_obj.run()
      File "/root/anaconda3/envs/acsz/lib/python3.7/site-packages/setuptools/command/install.py", line 61, in run
        return orig.install.run(self)
      File "/root/anaconda3/envs/acsz/lib/python3.7/distutils/command/install.py", line 545, in run
        self.run_command('build')
      File "/root/anaconda3/envs/acsz/lib/python3.7/distutils/cmd.py", line 313, in run_command
        self.distribution.run_command(command)
      File "/root/anaconda3/envs/acsz/lib/python3.7/distutils/dist.py", line 985, in run_command
        cmd_obj.run()
      File "/tmp/pip-install-zcc2vtwz/mitie/setup.py", line 36, in run
        subprocess.check_call(['make', 'mitielib'])
      File "/root/anaconda3/envs/acsz/lib/python3.7/subprocess.py", line 347, in check_call
        raise CalledProcessError(retcode, cmd)
    subprocess.CalledProcessError: Command '['make', 'mitielib']' returned non-zero exit status 2.
    ----------------------------------------
ERROR: Command errored out with exit status 1: /root/anaconda3/envs/acsz/bin/python -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-zcc2vtwz/mitie/setup.py'"'"'; __file__='"'"'/tmp/pip-install-zcc2vtwz/mitie/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' install --record /tmp/pip-record-pjnzlmaq/install-record.txt --single-version-externally-managed --compile Check the logs for full command output.
```

> 解决方法
```s
  yum groupinstall 'Development Tools'
```

### 安装 rasa_core

```python
    pip install rasa_core==0.9.0
```

> 注：
- tensorflow == 1.8.0
- keras == 2.1.5

## 项目初尝试

### 获取项目

```shell
  git clone https://github.com/zqhZY/_rasa_chatbot.git
```

### 项目目录介绍

```shell
  _rasa_chatbot/
  ├── bot.py
  ├── chat_detection
  ├── data
  │   ├── mobile_nlu_data.json # train data json format
  │   ├── mobile_raw_data.txt # train data raw
  │   ├── mobile_story.md # toy dialogue train data 
  │   └── total_word_feature_extractor.dat # pretrained mitie word vector
  ├── httpserver.py # rasa nlu httpserver
  ├── __init__.py
  ├── INSTALL.md
  ├── ivr_chatbot.yml # rasa nlu config file
  ├── mobile_domain.yml # rasa core config file
  ├── projects # pretrained models
  │   ├── dialogue
  │   └── ivr_nlu
  ├── README.md
  ├── tools # tools of data process
  └── train.sh # train script of rasa nlu
```

### 模型训练

```shell
  $ sh train.sh
  /home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/rasa_nlu-0.12.2-py3.6.egg/rasa_nlu/utils/__init__.py:236: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  2020-09-18 15:50:29.486477: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda-9.2/lib64:/usr/local/cuda-9.2/lib64:
  2020-09-18 15:50:29.486562: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda-9.2/lib64:/usr/local/cuda-9.2/lib64:
  2020-09-18 15:50:29.486576: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
  /home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/scipy/__init__.py:143: UserWarning: NumPy 1.14.5 or above is required for this version of SciPy (detected version 1.14.0)
    UserWarning)
  No Jieba Default Dictionary found
  No Jieba User Dictionary found
  Building prefix dict from the default dictionary ...
  Loading model from cache /tmp/jieba.cache
  Loading model cost 0.600 seconds.
  Prefix dict has been built successfully.
  2020-09-18 15:50:31 WARNING  rasa_nlu.extractors.mitie_entity_extractor - Example skipped: Invalid entity {'start': 5, 'end': 8, 'value': '这个月', 'entity': 'time'} in example '我想看一下这个月份的话费': entities must span whole tokens. Wrong entity end.
  2020-09-18 15:50:31 WARNING  rasa_nlu.extractors.mitie_entity_extractor - Example skipped: Invalid entity {'start': 7, 'end': 9, 'value': '三月', 'entity': 'time'} in example '你能给我看下我三月份消费多少了吗': entities must span whole tokens. Wrong entity end.
  2020-09-18 15:50:31 WARNING  rasa_nlu.extractors.mitie_entity_extractor - Example skipped: Invalid entity {'start': 7, 'end': 9, 'value': '三月', 'entity': 'time'} in example '你能给我看下我三月份的消费多少了吗': entities must span whole tokens. Wrong entity end.
  2020-09-18 15:50:31 WARNING  rasa_nlu.extractors.mitie_entity_extractor - Example skipped: Invalid entity {'start': 6, 'end': 8, 'value': '一月', 'entity': 'time'} in example '我想问一下我一月份是用了多少话费': entities must span whole tokens. Wrong entity end.
  2020-09-18 15:50:31 WARNING  rasa_nlu.extractors.mitie_entity_extractor - Example skipped: Invalid entity {'start': 7, 'end': 9, 'value': '三月', 'entity': 'time'} in example '帮我查一下我的三月份的消费多少': entities must span whole tokens. Wrong entity end.
  2020-09-18 15:50:31 WARNING  rasa_nlu.extractors.mitie_entity_extractor - Example skipped: Invalid entity {'start': 3, 'end': 5, 'value': '三月', 'entity': 'time'} in example '查下我三月份的用了多少钱': entities must span whole tokens. Wrong entity end.
  2020-09-18 15:50:31 WARNING  rasa_nlu.extractors.mitie_entity_extractor - Example skipped: Invalid entity {'start': 5, 'end': 7, 'value': '三月', 'entity': 'time'} in example '那个请问我三月份的一共消费多少': entities must span whole tokens. Wrong entity end.
  2020-09-18 15:50:31 WARNING  rasa_nlu.extractors.mitie_entity_extractor - Example skipped: Invalid entity {'start': 10, 'end': 12, 'value': '流量', 'entity': 'item'} in example '给我办一个三十的新流量业务': entities must span whole tokens. Wrong entity start.
  2020-09-18 15:50:31 WARNING  rasa_nlu.extractors.mitie_entity_extractor - Example skipped: Invalid entity {'start': 5, 'end': 7, 'value': '三十', 'entity': 'price'} in example '给我办一个三十的新流量业务': entities must span whole tokens. Wrong entity start.
  Training to recognize 4 labels: 'item', 'time', 'phone_number', 'price'
  Part I: train segmenter
  words in dictionary: 26649
  num features: 271
  now do training
  C:           20
  epsilon:     0.01
  num threads: 1
  cache size:  5
  max iterations: 2000
  loss per missed segment:  3
  C: 20   loss: 3 	0.927007
  C: 35   loss: 3 	0.927007
  C: 20   loss: 4.5 	0.934307
  C: 5   loss: 3 	0.934307
  C: 20   loss: 1.5 	0.919708
  C: 8.1457   loss: 5.41911 	0.941606
  C: 0.1   loss: 8.3092 	0.861314
  C: 0.1   loss: 4.67493 	0.861314
  C: 13.9759   loss: 5.8909 	0.941606
  C: 7.34113   loss: 5.70812 	0.941606
  C: 11.3072   loss: 5.48712 	0.934307
  C: 9.33113   loss: 5.3272 	0.941606
  C: 8.93262   loss: 5.24892 	0.941606
  C: 7.41157   loss: 5.3512 	0.941606
  C: 7.35558   loss: 5.4804 	0.941606
  best C: 8.1457
  best loss: 5.41911
  num feats in chunker model: 4095
  train: precision, recall, f1-score: 0.964539 0.992701 0.978417 
  Part I: elapsed time: 19 seconds.

  Part II: train segment classifier
  now do training
  num training samples: 142
  C: 200   f-score: 0.954142
  C: 400   f-score: 0.954142
  C: 300   f-score: 0.954142
  C: 100   f-score: 0.954142
  C: 0.01   f-score: 0.912356
  C: 600   f-score: 0.954142
  C: 1400   f-score: 0.954142
  C: 3000   f-score: 0.954142
  C: 5000   f-score: 0.954142
  C: 2550   f-score: 0.954142
  C: 1325   f-score: 0.954142
  C: 712.5   f-score: 0.954142
  C: 406.25   f-score: 0.954142
  C: 253.125   f-score: 0.954142
  C: 176.562   f-score: 0.961538
  C: 168.906   f-score: 0.954142
  C: 211.016   f-score: 0.954142
  C: 189.961   f-score: 0.954142
  C: 179.434   f-score: 0.954142
  C: 174.17   f-score: 0.954142
  C: 176.85   f-score: 0.961538
  C: 176.706   f-score: 0.961538
  C: 175.366   f-score: 0.954142
  C: 176.634   f-score: 0.961538
  C: 175.964   f-score: 0.954142
  best C: 176.562
  test on train: 
  82  0  0  0  0 
  0 50  0  0  0 
  0  0  2  0  0 
  0  0  0  3  0 
  0  1  0  0  4 

  overall accuracy: 0.992958
  Part II: elapsed time: 1336 seconds.
  df.number_of_classes(): 5
  Fitting 2 folds for each of 6 candidates, totalling 12 fits
  [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
  [Parallel(n_jobs=1)]: Done  12 out of  12 | elapsed:    0.3s finished
```

命令运行耗时较长，模型训练完毕生成：

```
projects/
└── ivr_nlu
    └── demo
        ├── entity_extractor.dat
        ├── entity_synonyms.json
        ├── intent_classifier_sklearn.pkl
        ├── metadata.json
        └── training_data.json
```

### 测试 rasa nlu

- 打开 第一个 窗口，运行 如下命令：

```shell
  $ python httpserver.py
  No Jieba Default Dictionary found
  No Jieba User Dictionary found
  2020-09-19 18:18:50+0800 [-] Log opened.
  2020-09-19 18:18:50+0800 [-] Site starting on 1235
  2020-09-19 18:18:50+0800 [-] Starting factory <twisted.web.server.Site object at 0x7f0a25b4c0f0>

```

- 打开第二个 窗口，运行如下命令：

```shell
  $ curl -X POST localhost:1235/parse -d '{"q":"我的流量还剩多少"}' | python -m json.tool
    % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                  Dload  Upload   Total   Spent    Left  Speed
  100   125    0    93  100    32    138     47 --:--:-- --:--:-- --:--:--   138
  {
      "q": "\u6211\u7684\u6d41\u91cf\u8fd8\u5269\u591a\u5c11",
      "intent": "request_search",
      "entities": {
          "item": "\u6d41\u91cf"
      }
  }
```
> 由于 该编码 为 utf-8 编码，所以 看不懂

- 第一个窗口

```shell
  2020-09-19 18:20:00+0800 [-] {'q': '我的流量还剩多少'}
  Building prefix dict from the default dictionary ...
  Loading model from cache /tmp/jieba.cache
  Loading model cost 0.664 seconds.
  Prefix dict has been built successfully.
  2020-09-19 18:20:00+0800 [-] /home/amy/.conda/envs/rasa/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: builtins.DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
  2020-09-19 18:20:00+0800 [-] {'intent': {'name': 'request_search', 'confidence': 0.7187524752189071}, 'entities': [{'entity': 'item', 'value': '流量', 'start': 2, 'end': 4, 'confidence': None, 'extractor': 'ner_mitie'}], 'intent_ranking': [{'name': 'request_search', 'confidence': 0.7187524752189071}, {'name': 'request_management', 'confidence': 0.0575744190015811}, {'name': 'deny', 'confidence': 0.03970444098955579}, {'name': 'confirm', 'confidence': 0.036774399374054405}, {'name': 'inform_time', 'confidence': 0.028818810951793845}, {'name': 'inform_item', 'confidence': 0.022871165137948027}, {'name': 'inform_current_phone', 'confidence': 0.020024429738488236}, {'name': 'greet', 'confidence': 0.018917055091065192}, {'name': 'unknown_intent', 'confidence': 0.01426294634562408}, {'name': 'inform_package', 'confidence': 0.013179727083893375}], 'text': '我的流量还剩多少'}
  2020-09-19 18:20:00+0800 [-] {'q': '我的流量还剩多少', 'intent': 'request_search', 'entities': {'item': '流量'}}
```

### 训练对话

```shell
$ python bot.py train-dialogue
No Jieba Default Dictionary found
No Jieba User Dictionary found
Using TensorFlow backend.
/home/amy/.conda/envs/rasa/lib/python3.6/site-packages/rasa_core/utils.py:341: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  return yaml.load(read_file(filename, "utf-8"))
INFO:rasa_nlu.components:Added 'nlp_mitie' to component cache. Key 'nlp_mitie-/web/workspace/yangkm/python_wp/nlu/DSWp/_rasa_chatbot-master/data/total_word_feature_extractor.dat'.
No Jieba Default Dictionary found
No Jieba User Dictionary found
Processed Story Blocks: 100%|████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 1410.32it/s, # trackers=1]
Processed Story Blocks: 100%|█████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 175.66it/s, # trackers=7]
Processed Story Blocks: 100%|█████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 43.55it/s, # trackers=20]
Processed actions: 196it [00:00, 9426.05it/s, # examples=174]
INFO:rasa_core.policies.memoization:Memorized 174 unique action examples.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
masking_1 (Masking)          (None, 5, 31)             0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 32)                8192      
_________________________________________________________________
dense_1 (Dense)              (None, 11)                363       
_________________________________________________________________
activation_1 (Activation)    (None, 11)                0         
=================================================================
Total params: 8,555
Trainable params: 8,555
Non-trainable params: 0
_________________________________________________________________
INFO:rasa_core.policies.keras_policy:Fitting model with 196 total samples and a validation split of 0.2
Train on 156 samples, validate on 40 samples
Epoch 1/200
2020-09-19 18:23:07.850568: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
156/156 [==============================] - 1s 9ms/step - loss: 2.3155 - acc: 0.3077 - val_loss: 2.2575 - val_acc: 0.3750
...
Epoch 200/200
156/156 [==============================] - 0s 569us/step - loss: 0.1467 - acc: 0.9423 - val_loss: 0.8500 - val_acc: 0.7250
INFO:rasa_core.policies.keras_policy:Done fitting keras policy model
INFO:rasa_core.agent:Model directory projects/dialogue exists and contains old model files. All files will be overwritten.
INFO:rasa_core.agent:Persisted model to '/web/workspace/yangkm/python_wp/nlu/DSWp/_rasa_chatbot-master/projects/dialogue'
```
- 模型训练完毕生成：

```shell
projects
├── dialogue
│   ├── domain.json
│   ├── domain.yml
│   ├── policy_0_MemoizationPolicy
│   │   ├── featurizer.json
│   │   └── memorized_turns.json
│   ├── policy_1_KerasPolicy
│   │   ├── featurizer.json
│   │   ├── keras_arch.json
│   │   ├── keras_policy.json
│   │   └── keras_weights.h5
│   ├── policy_metadata.json
│   └── stories.md
└── ivr_nlu
```

### 在线模式下的对话训练

利用 强化学习 的 方法，生成训练集，可以视为 交互式人工标注工具

```shell
  $ python bot.py online_train
```

- output:

```
INFO:rasa_nlu.components:Added 'nlp_mitie' to component cache. Key 'nlp_mitie-/web/workspace/yangkm/python_wp/nlu/DSWp/_rasa_chatbot-master/data/total_word_feature_extractor.dat'.
No Jieba Default Dictionary found
No Jieba User Dictionary found
Processed Story Blocks: 100%|████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 2510.27it/s, # trackers=1]
Processed Story Blocks: 100%|█████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 325.58it/s, # trackers=7]
Processed Story Blocks: 100%|█████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 78.36it/s, # trackers=20]
Processed actions: 196it [00:00, 15899.50it/s, # examples=174]
INFO:rasa_core.policies.memoization:Memorized 174 unique action examples.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
masking_1 (Masking)          (None, 5, 31)             0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 32)                8192      
_________________________________________________________________
dense_1 (Dense)              (None, 11)                363       
_________________________________________________________________
activation_1 (Activation)    (None, 11)                0         
=================================================================
Total params: 8,555
Trainable params: 8,555
Non-trainable params: 0
_________________________________________________________________
INFO:rasa_core.policies.keras_policy:Fitting model with 196 total samples and a validation split of 0.0
Epoch 1/200
2020-09-19 18:30:48.658524: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
196/196 [==============================] - 0s 2ms/step - loss: 2.2901 - acc: 0.2857
...
Epoch 200/200
196/196 [==============================] - 0s 299us/step - loss: 0.1796 - acc: 0.9235
INFO:rasa_core.policies.keras_policy:Done fitting keras policy model
Bot loaded. Type a message and press enter: 
你好
Building prefix dict from the default dictionary ...
DEBUG:jieba:Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
DEBUG:jieba:Loading model from cache /tmp/jieba.cache
Loading model cost 0.622 seconds.
DEBUG:jieba:Loading model cost 0.622 seconds.
Prefix dict has been built successfully.
DEBUG:jieba:Prefix dict has been built successfully.
/home/amy/.conda/envs/rasa/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
  if diff:
------
Chat history:

	bot did:	None

	bot did:	action_listen

	user said:	我想看一下消费情况

		 whose intent is:	request_search

	with item:	消费

we currently have slots: item: 消费, phone_number: None, price: None, time: None

------
The bot wants to [utter_ask_time] due to the intent. Is this correct?

	1.	Yes
	2.	No, intent is right but the action is wrong
	3.	The intent is wrong
	0.	Export current conversations as stories and quit
1
你想查哪个时间段的
------
Chat history:

	bot did:	None

	bot did:	action_listen

	user did:	request_search

	with item:	消费

	bot did:	utter_ask_time

we currently have slots: item: 消费, phone_number: None, price: None, time: None

------
The bot wants to [action_listen]. Is this correct?

	1.	Yes.
	2.	No, the action is wrong.
	0.	Export current conversations as stories and quit
十二月分份
Invalid answer, only 1, 2, 0 allowed

	1.	Yes.
	2.	No, the action is wrong.
	0.	Export current conversations as stories and quit
1
Next user input:
一月
/home/amy/.conda/envs/rasa/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
  if diff:
------
Chat history:

	bot did:	action_listen

	user did:	request_search

	with item:	消费

	bot did:	utter_ask_time

	bot did:	action_listen

	user said:	一月

		 whose intent is:	inform_time

	with time:	一月

we currently have slots: item: 消费, phone_number: None, price: None, time: 一月

------
The bot wants to [action_search_consume] due to the intent. Is this correct?

	1.	Yes
	2.	No, intent is right but the action is wrong
	3.	The intent is wrong
	0.	Export current conversations as stories and quit
1
好，请稍等
您好，您一月共消费二十八元。
------
Chat history:

	bot did:	utter_ask_time

	bot did:	action_listen

	user did:	inform_time

	with time:	一月

	bot did:	action_search_consume

we currently have slots: item: 消费, phone_number: None, price: None, time: 一月

------
The bot wants to [utter_ask_morehelp]. Is this correct?

	1.	Yes.
	2.	No, the action is wrong.
	0.	Export current conversations as stories and quit
1
还有什么能帮您吗

```


### 测试 对话 功能

```shell
  $ python bot.py run
```

- 场景一：

```shell
  Bot loaded. Type a message and press enter: 
  你是谁
  Building prefix dict from the default dictionary ...
  DEBUG:jieba:Building prefix dict from the default dictionary ...
  Loading model from cache /tmp/jieba.cache
  DEBUG:jieba:Loading model from cache /tmp/jieba.cache
  Loading model cost 0.622 seconds.
  DEBUG:jieba:Loading model cost 0.622 seconds.
  Prefix dict has been built successfully.
  DEBUG:jieba:Prefix dict has been built successfully.
  /home/amy/.conda/envs/rasa/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
    if diff:
  你好!，我是小热，可以帮您办理流量套餐，话费查询等业务。
  我想看一下消费情况
  /home/amy/.conda/envs/rasa/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
    if diff:
  你想查几月份的
  一月份的
  /home/amy/.conda/envs/rasa/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
    if diff:
  好，请稍等
  您好，您一月份共消费二十八元。
  还有什么能帮您吗
  好的
  /home/amy/.conda/envs/rasa/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
    if diff:
  还有什么能帮您吗
  学习
  /home/amy/.conda/envs/rasa/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
    if diff:
  Bye， 下次再见
```


## 项目核心内容分析

### ivr_chatbot.yml 配置文件 分析

```
  language: "zh"                                      # 设置 为 中文模式
  project: "ivr_nlu"
  fixed_model_name: "demo"
  path: "models"
  pipeline:
  - name: "nlp_mitie"
    model: "data/total_word_feature_extractor.dat"
  - name: "tokenizer_jieba"
    default_dict: "./default_dict.big"
    user_dicts: "./jieba_userdict"
#   user_dicts: "./jieba_userdict/jieba_userdict.txt"
  - name: "ner_mitie"
  - name: "ner_synonyms"
  - name: "intent_entity_featurizer_regex"
  - name: "intent_featurizer_mitie"
  - name: "intent_classifier_sklearn"
```

- 使用 中文模式，需要采用以下 两个 pipelines:
  - 使用 MITIE+Jieba: [“nlp_mitie”, “tokenizer_jieba”, “ner_mitie”, “ner_synonyms”, “intent_classifier_mitie”]
    - 特点：这种方式训练比较慢，效果也不是很好，最后出现的intent也没有分数排序
  - MITIE+Jieba+sklearn (sample_configs/config_jieba_mitie_sklearn.json): [“nlp_mitie”, “tokenizer_jieba”, “ner_mitie”, “ner_synonyms”, “intent_featurizer_mitie”, “intent_classifier_sklearn”]
    - 流程介绍：
      - ”nlp_mitie”初始化MITIE；
      - ”tokenizer_jieba”用jieba来做分词；
      - ”ner_mitie”和”ner_synonyms”做实体识别；
      - ”intent_featurizer_mitie”为意图识别做特征提取；
      - ”intent_classifier_sklearn”使用sklearn做意图识别的分类；

- 使用 用户自定义词典
  - default_dict ： 默认字典
  - user_dicts： 用户自定义词典

### ./data/mobile_nlu_data.json 意图识别和实体识别 训练数据 分析

- 通用 句式 common_examples：只要满足 句式，就匹配结果
  - 格式转化方法 tools/trainsfer_raw_to_rasa.py 
    - 原始格式：./data/mobile_raw_data.txt
    - 转换后格式：./data/mobile_nlu_data.json
- 正则表达式 regex_features：利用 正则表达式 匹配 句式
- 同义词匹配 entity_synonyms：同义词 匹配

```shell
  {
  "rasa_nlu_data": {
    "common_examples": [
      {
        "text": "帮我查一下我上个月的流量有多少",
        "intent": "request_search",
        "entities": [
          {
            "start": 10,
            "end": 12,
            "value": "流量",
            "entity": "item"
          },...
        ]
      },...
    ],
    "regex_features": [
      {
        "name": "inform_package",
        "pattern": "套餐[0-9一二三四五六七八九十百俩两]+"
      },
      {
        "name": "inform_time",
        "pattern": "([0-9一二三四五六七八九十百俩两]+)月份?的?"
      }
    ],
    "entity_synonyms": [{
        "value": "消费",
        "synonyms": ["话费", "钱"]
      }]
  }
}

```

### mobile_story.md 对话设置 文件 分析

- 内容类型：训练数据，对话流程作为训练数据训练policy 决策模块（内容为 unicode 编码，可以对照 unicode2zh.md 文件查看 翻译，或用 [Unicode编码转换](https://tool.chinaz.com/tools/unicode.aspx)翻译工具）
- 介绍：教会你的助手如何回复你的信息。这称为对话管理(dialogue management)，由你的Core模型来处理。Core模型以训练“故事”的形式从真实的会话数据中学习。故事是用户和助手之间的真实对话。带有意图和实体的行为反映了用户的输入和操作名称，操作名称展示了助手应该如何响应。 

```
  ## Generated Story 5914322956106259965
  > check_asked_question
  * greet
      - utter_greet
  * request_search{"item": "\u7684\u60c5\u51b5"}
      - slot{"item": "\u6d88\u8d39"}
      - slot{"item": "\u7684\u60c5\u51b5"}
      - action_search_consume
  * request_search{"item": "\u6d88\u8d39"}
      - slot{"item": "\u6d88\u8d39"}
      - action_search_consume
  * inform_time{"time": "\u4e0a\u4e2a\u6708"}
      - slot{"time": "\u4e0a\u4e2a\u6708"}
      - action_search_consume
      - utter_ask_morehelp
  * deny
      - utter_goodbye
      - export
  * affirm OR thankyou
      – action_handle_affirmation
  ...
```
> 以 - 开头的行是助手所采取的操作，所有的操作都是发送回用户的消息，比如utter_greet;
> 以 * 开头的行是 用户意图

- 三部分：
  - 用户输入（User Messages）
    - 介绍：使用 **“*”开头的语句表示用户的输入消息**，我们无需使用包含某个具体内容的输入，而是**使用NLU管道输出的intent和entities来表示可能的输入**。需要注意的是，如果用户的输入可能包含entities，建议将其包括在内，将有助于policies预测下一步action。
    - 举例：
      - * greet 表示用户输入没有entity情况；
      - * inform_time{"time": "\u4e0a\u4e2a\u6708"} 表示用户输入包含entity情况，响应这类intent为普通action；
      - * request_search{"item": "\u6d88\u8d39"} 表示用户输入Message对应的intent为form action情况；
  - 动作（Actions）
    - 介绍：使用 “-” 开头的语句表示要执行动作(Action)；
    - 类别：
      - utterance actions：在domain.yaml中定义以utter_为前缀，比如名为greet的意图，它的回复应为utter_greet；
      - custom actions：自定义动作，具体逻辑由我们自己实现，虽然在定义action名称的时候没有限制，但是还是建议以action_为前缀，比如名为inform的意图fetch_profile的意图，它的response可为action_fetch_profile；
  - 事件（Events）
    - 介绍： 使用 “-” 开头，主要包含槽值设置(SlotSet)和激活/注销表单(Form)，它是是Story的一部分，并且必须显示的写出来。
    - 分类：
      - Slot Events：当我们在自定义Action中设置了某个槽值，那幺我们就需要在Story中Action执行之后显着的将这个SlotSet事件标注出来，格式为- slot{“slot_name”: “value”}。
      - Form Events：在Story中主要存在三种形式的表单事件(Form Events)，它们可表述为：
        - Form Action事件 ：单动作事件，是自定义Action的一种，用于一个表单操作；
        - Form activation事件：激活表单事件，当form action事件执行后，会立马执行该事件；
        - Form deactivation事件：注销表单事件，作用与form activation相反；
  - > check_asked_question：表示first story，这样在其他story中，如果有相同的first story部分，可以直接用> check_asked_question代替；
  - OR Statements：主要用于实现某一个action可同时响应多个意图的情况

> 注：参考 [Rasa中文聊天机器人开发指南(3)：Core篇](https://blog.csdn.net/AndrExpert/article/details/105434136)

### mobile_domain.yml 定义域 文件 分析

- 内容：定义域，描述了对话机器人应知道的所有信息，类似于“人的大脑”，存储了意图intents、实体entities、插槽slots以及动作actions等信息，其中，intents、entities在NLU训练样本中定义，slots对应于entities类型，只是表现形式不同。
- 介绍：域定义了助手所处的环境:
  - 有哪些 slot;
  - 它应该期望得到什么用户输入;
  - 它应该能够预测什么操作;
  - 系统可能采用哪些 action;
  - 存储什么信息；
- 类别介绍：
  - intents：描述 Bot 拥有 哪些意图识别 的能力；
  - sesstion_config：描述一次会谈的超时时间和超时后进行下一次会谈的行为
    - session_expiration_time: 60 -> 每次会话的超时时间为60s，如果用户开始一段会话后，在60s内没有输入任何信息，那幺这次会话将被结束，然后Bot又会开启一次新的会话，并将上一次会话的Slot值拷贝过来;
    - carry_over_slots_to_new_session: true -> 是否将 上一次会话Slot的值 copy 到新会谈；
  - slots：定义 Bot 拥有哪些 槽值能力；
  - entities：定义 Bot 拥有识别哪些 实体的能力，类似于输入文本中的关键字，需要在NLU样本中进行标注，然后Bot进行实体识别，并将其填充到Slot槽中；
  - actions：定义 Bot 有哪些可执行的行为。
    - 介绍：当Rasa NLU识别到用户输入Message的意图后，Rasa Core对话管理模块就会对其作出回应，而完成这个回应的模块就是action。
    - 类别：
      - default actions：是Rasa Core默认的一组actions，我们无需定义它们，直接可以story和domain中使用。
        - action_listen：监听action，Rasa Core在会话过程中通常会自动调用该action；
        - action_restart：重置状态，比初始化Slots(插槽)的值等；
        - action_default_fallback：当Rasa Core得到的置信度低于设置的阈值时，默认执行该action；
      - utter actions：是以utter_为开头，仅仅用于向用户发送一条消息作为反馈的一类actions。定义一个UtterAction很简单，只需要在domain.yml文件中的actions:字段定义以utter_为开头的action即可，而具体回复内容将被定义在templates:部分
      - custom actions：即自定义action，允许开发者执行任何操作并反馈给用户，比如简单的返回一串字符串，或者控制家电、检查银行账户余额等等。它与DefaultAction不同，自定义action需要我们在domain.yml文件中的actions部分先进行定义，然后在指定的webserver中实现它，其中，这个webserver的url地址在endpoint.yml文件中指定，并且这个webserver可以通过任何语言实现，当然这里首先推荐python来做，毕竟Rasa Core为我们封装好了一个rasa-core-sdk专门用来处理自定义action。关于action web的搭建和action的具体实现。
  - forms：定义 Bot 有哪些 form action；
  - responses：定义当触发 某个意图后， Bot 能够使用其中的文本自动回复；

> 注：参考 [Rasa中文聊天机器人开发指南(3)：Core篇](https://blog.csdn.net/AndrExpert/article/details/105434136)
```
  # 槽位，对应于entities类型
  slots:                      
    item:
      type: text
    time:
      type: text
    phone_number:
      type: text
    price:
      type: text

  # 用户意图
  intents:
    - greet
    - confirm
    - goodbye
    - thanks
    - inform_item
    - inform_package
    - inform_time
    - request_management
    - request_search
    - deny
    - inform_current_phone
    - inform_other_phone

  entities:
    - item
    - time
    - phone_number
    - price

  # NLG：基于模板，每个模板 对应 多条回复，每次回复 可从中 中随机挑选 一条 返回
  templates:
    utter_greet:
      - "您好!，我是机器人小热，很高兴为您服务。"
      - "你好!，我是小热，可以帮您办理流量套餐，话费查询等业务。"
      - "hi!，人家是小热，有什么可以帮您吗。"
    utter_goodbye:
      - "再见，为您服务很开心"
      - "Bye， 下次再见"
    utter_default:
      - "您说什么"
      - "您能再说一遍吗，我没听清"
    utter_thanks:
      - "不用谢"
      - "我应该做的"
      - "您开心我就开心"
    utter_ask_morehelp:
      - "还有什么能帮您吗"
      - "您还想干什么"
    utter_ask_time:
      - "你想查哪个时间段的"
      - "你想查几月份的"
    utter_ask_package:
      - "我们现在支持办理流量套餐：套餐一：二十元包月三十兆；套餐二：四十元包月八十兆，请问您需要哪个？"
      - "我们有如下套餐供您选择：套餐一：二十元包月三十兆；套餐二：四十元包月八十兆，请问您需要哪个？"
    utter_ack_management:
      - "已经为您办理好了{item}"   #  {item} 模板填充——实际中应该去查库得到

  # 机器反应，每个 action 对应 一个模板
  actions:
    - utter_greet
    - utter_goodbye
    - utter_default
    - utter_thanks
    - utter_ask_morehelp
    - utter_ask_time
    - utter_ask_package
    - utter_ack_management
    - bot.ActionSearchConsume    # 可能还有查库的需求——> 自定义 action
  forms:
    – weather_form
  responses:
    utter_answer_greet:
      – text: “您好！请问我可以帮到您吗？”
      – text: “您好！很高兴为您服务。请说出您要查询的功能？”
    utter_default:
      – text: “没听懂，请换种说法吧~”


```

## 参考资料

1. [_rasa_chatbot](https://github.com/zqhZY/_rasa_chatbot)
2. [Rasa中文聊天机器人开发指南(1)：入门篇](https://jiangdg.blog.csdn.net/article/details/104328946)
3. [Rasa中文聊天机器人开发指南(2)：NLU篇](https://jiangdg.blog.csdn.net/article/details/104530994)
4. [Rasa中文聊天机器人开发指南(3)：Core篇](https://blog.csdn.net/AndrExpert/article/details/105434136)
5. [Rasa 安装](http://rasachatbot.com/2_Rasa_Tutorial/#rasa)
6. [Rasa 学习](https://blog.csdn.net/ljp1919/category_9656007.html)
7. [rasa_chatbot_cn](https://github.com/GaoQ1/rasa_chatbot_cn)
8.  [用Rasa NLU构建自己的中文NLU系统](http://www.crownpku.com/2017/07/27/用Rasa_NLU构建自己的中文NLU系统.html)
9.  [Rasa_NLU_Chi](https://github.com/crownpku/Rasa_NLU_Chi)
10. [rasa 源码分析](https://www.zhihu.com/people/martis777/posts)
