# 【关于 python 】 的那些你不知道的事


> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 目录

- [【关于 python 】 的那些你不知道的事](#关于-python--的那些你不知道的事)
  - [目录](#目录)
  - [*args 和 **kwargs的用法](#args-和-kwargs的用法)
    - [动机：](#动机)
    - [用途：](#用途)
    - [*args：](#args)
    - [**kwargs:](#kwargs)
    - [对比 *args 与 **kwargs](#对比-args-与-kwargs)
    - [参考资料](#参考资料)
  - [装饰器](#装饰器)
    - [装饰器基本介绍](#装饰器基本介绍)
    - [实践](#实践)
    - [参考资料](#参考资料-1)


## *args 和 **kwargs的用法

### 动机：

对于一些编写的函数，可能预先并不知道, 函数使用者会传递多少个参数给你, 所以在这个场景下使用这两个关键字。
### 用途：

*args 和 **kwargs 主要用于函数定义。 你可以将不定数量的参数传递给一个函数；

### *args：
- 介绍：用来发送一个非键值对的可变数量的参数列表给一个函数；
- 举例
```s
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)

test_var_args('yasoob', 'python', 'eggs', 'test')

# output
first normal arg: yasoob
another arg through *argv: python
another arg through *argv: eggs
another arg through *argv: test

```

### **kwargs:
- 介绍：允许你将不定长度的键值对, 作为参数传递给一个函数。 如果你想要在一个函数里处理带名字的参数, 你应该使用kwargs;
- 举例：
```s
def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} == {1}".format(key, value))

# output
>>> greet_me(name="yasoob")
name == yasoob

```
### 对比 *args 与 **kwargs

```s
def test_args_kwargs(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)

```
> 使用 *args
```s
>>> args = ("two", 3, 5)
>>> test_args_kwargs(*args)
arg1: two
arg2: 3
arg3: 5
```

> 使用 **kwargs
```s
>>> kwargs = {"arg3": 3, "arg2": "two", "arg1": 5}
>>> test_args_kwargs(**kwargs)
arg1: 5
arg2: two
arg3: 3
```
### 参考资料
1. *args 和 **kwargs的用法 : https://www.jianshu.com/p/d993b2a88e73

## 装饰器

### 装饰器基本介绍

- 装饰器本质：一个 Python 函数或类；
- 作用：可以让其他函数或类在不需要做任何代码修改的前提下增加额外功能，装饰器的返回值也是一个函数/类对象。
- 使用场景：经常用于有切面需求的场景，比如：插入日志、性能测试、事务处理、缓存、权限校验等场景，装饰器是解决这类问题的绝佳设计。
- 优点：有了装饰器，我们就可以抽离出大量与函数功能本身无关的雷同代码到装饰器中并继续重用。概括的讲，装饰器的作用就是为已经存在的对象添加额外的功能。

### 实践

> 简单装饰器
```s
def use_logging(func):

    def wrapper():
        logging.warn("%s is running" % func.__name__)
        return func()   # 把 foo 当做参数传递进来时，执行func()就相当于执行foo()
    return wrapper

def foo():
    print('i am foo')

foo = use_logging(foo)  # 因为装饰器 use_logging(foo) 返回的时函数对象 wrapper，这条语句相当于  foo = wrapper
foo()                   # 执行foo()就相当于执行 wrapper()
```
> @ 语法糖
```s
def use_logging(func):

    def wrapper():
        logging.warn("%s is running" % func.__name__)
        return func()
    return wrapper

@use_logging
def foo():
    print("i am foo")

foo()
```
> 带参数的装饰器
```s
# 功能：加载数据
def loadData(filename):
    '''
        功能：加载数据
        input:
            filename   String 文件名称 
        return:
            data       List    数据列表
    '''
    data = []
    with open(filename,"r",encoding="utf-8") as f:
        line = f.readline().replace("\n","")
        while line:
            data.append(line)
            line = f.readline().replace("\n","")
    return data

# 功能：装饰器 之 数据采样
def simpleData(func):
    '''
        功能：装饰器 之 数据采样
    '''
    def wrapper(*args):
        dataList = func(*args)
        rate = 0.05
        dataListLen = len(dataList)
        if dataListLen>100000:
            rate = 0.001
        elif dataListLen>10000:
            rate = 0.01
        elif dataListLen>1000:
            rate = 0.05
        elif dataListLen>100:
            rate = 0.1
        else:
            rate =1
        shuffle(dataList)
        simpleDataList =dataList[:int(rate*len(dataList))]
        return dataList,simpleDataList
    return wrapper

# 使用
dataList,simpleDataList = simpleData(loadData)(f"{basePath}{name}.txt")

```

### 参考资料

1. 理解 Python 装饰器看这一篇就够了:https://foofish.net/python-decorator.html


