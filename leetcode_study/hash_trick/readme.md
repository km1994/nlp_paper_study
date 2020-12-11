# 【关于 Leetcode 刷题篇 之 哈希表总结】那些你不知道的事

> 作者：杨夕 <br/>
> 项目地址：https://github.com/km1994/leetcode/tree/master/topic8_binary_search <br/>
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在刷题过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。 <br/>

## 动机

- 场景：

> 假设 你要 读取 一个 文件中，然后查询里面是否 某个数，请问你会采用的方法是什么？

> 假设 从 区间 [1,200] 内查询 某个数

![](img/20201211155608.png)

> 假设 从 区间 [1,100000] 内查询 某个数

![](img/20201211155633.png)

- 介绍：可以看出在小数据集下，列表和哈希表查询效率差异不大，但是当 数据 上万时，列表的查询效率将 变得低效。

## 解析

### 列表

- 其结构如下：

![](img/20201211160329.png)

- 查询方式：从头一个个 比较，直到匹配到结果，返回结果，否则返回 False；
- 查询复杂度：
  - 时间：O(N)

### 哈希表

- 方法：使用某种算法操作(散列函数)将键转化为数组的索引来访问数组中的数据，这样可以通过Key-value的方式来访问数据，达到常数级别的存取效率。
- 其结构如下【以拉链法为例】：

![](img/20201211160757.png)

- 查询复杂度：
  - 时间：O(1)

## 应用场景

### 应用场景一 提供更多信息

#### 介绍

- 场景介绍：题目要求返回更多的信息，比如对于一个 数组，你不仅需要判断 给定值 是否 存在于 数组中，你还需要返回其对应的索引
- 解决方式：利用 哈希表 建立 值 到 键 间的映射关系

#### 代码示例

```s
    # 读取大文件
    large_file = "large.txt"
    large_list = []
    large_dict = dict()
    with open(large_file,"r",encoding="utf-8") as f:
        line = f.readline()
        while line:
            large_list.append(int(line))
            large_dict[int(line)] = len(large_list) 
            line = f.readline()

    key = 3000
    # 利用 列表 查询 3000 的 索引
    for i in range(len(large_list)):
        if key == large_list[i]:
            print(f"Use list select {key} key is {i+1}")
            break

    # 利用 哈希表 查询 3000 的 索引
    print(f"Use Dict select {key} key is {large_dict[key]}")
```

#### 例题讲解

##### leetCode 205. 同构字符串

- 题目：
  
给定两个字符串 s 和 t，判断它们是否是同构的。

如果 s 中的字符可以被替换得到 t ，那么这两个字符串是同构的。

所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。

示例 1:

```s
    输入: s = "egg", t = "add"
    输出: true
```

示例 2:
```s
    输入: s = "foo", t = "bar"
    输出: false
```

示例 3:
```s
    输入: s = "paper", t = "title"
    输出: true
```

说明:你可以假设 s 和 t 具有相同的长度。

- 解析

定义一个 哈希表 dic 用于 存储 s 中每个字符 所对应 的 t 中 对应位置的字符

- 代码

```s
class Solution():
    def isIsomorphic(self,s,t):
        s_len = len(s)
        t_len = len(t)
        if s_len!=t_len:
            return False
        dic = {}
        for i in range(s_len):
            if s[i] in dic and dic[s[i]]!=t[i]:
                return False
            else:
                dic[s[i]]=t[i]
        return True 

solution = Solution()
print(f"egg and add is Isomorphic:{solution.isIsomorphic('egg','add')}")   
print(f"foo and bar is Isomorphic:{solution.isIsomorphic('foo','bar')}")   
print(f"paper and title is Isomorphic:{solution.isIsomorphic('paper','title')}")   
```

### 应用场景二 按键聚合或计数

#### 介绍

- 场景介绍：对于一个 数组，我们需要找出 其中 只出现过一次的 数值
- 解决方式：利用 哈希表 对 数组中每一个 数值 进行计数，最后返回 只才能一次的 数值

#### 例题讲解

##### leetCode 387. 字符串中的第一个唯一字符

- 题目

给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

案例:

```s
    s = "leetcode"
    返回 0.

    s = "loveleetcode",
    返回 2.
```

- 解析

利用 哈希表 对 数组中每一个 数值 进行计数，最后返回 只才能一次的 数值

- 代码

```s
class Solution2():
    def firstUniqChar(self,s):
        dic = {}
        for c in s:
            if c not in dic:
                dic[c]=0
            dic[c] +=1
        for i, c in enumerate(s):
            if dic[c] == 1:
                return i
        return -1

solution = Solution2()
print(f"leetcode the firstUniqChar:{solution.firstUniqChar('leetcode')}")   
print(f"loveleetcode the firstUniqChar:{solution.firstUniqChar('loveleetcode')}") 
```

## 参考资料

1. [leetCode. 哈希表专题(3)](https://zhuanlan.zhihu.com/p/58938611)