# 你还在为如何搞科研而发愁么？

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

## 目录

- 你曾经是否也遇到和我一样的问题？
  - 作为一名 scholar，你是否和我一样，在刚入门 NLP 时，对于陌生领域有种无从下手，心存畏惧？
  - 作为一名 scholar，你是否还在发愁如何找好的论文？
  - 作为一名 scholar，你是否还在为 自己 的 英文阅读 能力跟不上 很烦恼？
  - 作为一名 scholar，你是否还在为 看到 一篇好paper，但是复现不出 code 而心累？
  - 作为一名 scholar，你是否还在为 有Good idea，Outstanding Experimental results，Beautiful Chinese manuscript，结果 Bad English manuscript, Poor Journal 而奔溃？
  - 作为一名 scholar，你是否在为搞科研没人交流而自闭？
- 当你看到这一篇文档，你将不在为这些问题而烦恼，因为我们为你准备了一整套免费的从 论文查找->论文翻译->论文理解->相关代码搜索->写英文稿->科研学术交流 的路径。
  - [论文不会找怎么办？](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#论文不会找怎么办)
    - [顶会资讯](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#顶会资讯)
    - [论文搜索和分析工具](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#论文搜索和分析工具)
  - [外文读不懂怎么办？](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#外文读不懂怎么办)
    - [论文翻译神器 ———— 通天塔](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#论文翻译神器--通天塔)
    - [论文翻译小助手 ———— 彩云小译](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#论文翻译小助手--彩云小译)
  - [外文没 code 怎么办？](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#外文没-code-怎么办)
    - [papers with code](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#papers-with-code) 
    - [OpenGitHub 新项目快报](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#opengithub-新项目快报) 
  - [外文写起来麻烦怎么办](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#外文写起来麻烦怎么办) 
    - [Overleaf](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#overleaf) 
    - [Authorea](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#authorea) 
    - [Code ocean](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#code-ocean) 
  - [搞科研没人交流怎么办？](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#搞科研没人交流怎么办) 
    - [Shortscience](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#shortscience) 
    - [OpenReview](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#openreview) 
    - [Scirate](https://github.com/km1994/nlp_paper_study/tree/master/论文学习idea#scirate) 

## 论文不会找怎么办？

### 顶会资讯

一般不同方向都有该方向对应的顶会论文。

1. 机器学习(Machine Learning)百花齐放系列:

    机器学习作为人工智能的一个分支，涉及到概率论、统计学、计算复杂性理论等多门学科，目的是让计算机能像人一样自动学习，从而不断完善自身性能。机器学习的技术更新可广泛应用于计算机视觉、自然语言处理、搜索引擎、数据挖掘等领域。机器学习领域主要有两大国际会议最具影响力，分别是ICML和NIPS。

 - [ICML](https://icml.cc/Conferences/2019/) 

    ICML：International Conference on Machine Learning

    国际机器学习大会，由IMLS国际机器学习协会支持，始于1980年，此后每年夏季举行，2014年举办地在北京。ICML会接收到各路机器学习大牛的投稿，录用率只有百分之二十多。第36届ICML将于2019年6月10日到15日在美国长滩举办。

 - [NIPS](https://neurips.cc/Conferences/2019) 

    NIPS：Neural Information Processing Systems

    神经信息处理系统会议由NIPS基金会主办，固定在每年12月举行，是机器学习和计算神经科学领域的顶级会议，最初是为研究生物和人造神经网络的研究者设计的交叉学科会议，所以早期的会议主题既包括如何解决纯粹的工程问题，也包括利用机器模型理解生物神经网络。

    后来生物系统的研究和人造系统的研究分道扬镳，NIPS的会议主题就主要集中在了机器学习、人工智能和统计学上。

2. AI人工智能顶会系列:

 - [AAAI](https://aaai.org/Conferences/AAAI-19/) 

    AAAI: AAAI Conference on Artificial Intelligence

    AAAI会议由人工智能促进协会AAAI（Association for the Advancement of Artificial Intelligence）主办。人工智能促进协会是一个国际化的非营利科学组织，旨在推动人工智能领域的研究和应用，增进大众对人工智能的了解。AAAI会议始于1980年，既注重理论，也注重应用，还会讨论对人工智能发展有着重要影响的社会、哲学、经济等话题。

 - [IJCAI](https://ijcai19.org/) 

    IJCAI: International Joint Conferences on AI

    IJCAI是一家1969年于美国加州成立的非营利企业，致力于推动科学和教育的发展，由负责会议和期刊的两个部分组成。IJCAI会议始于1969年，每两年举办一次，从2016年开始改为一年举办一次。IJCAI会议选拔要求非常严格，论文录取率几乎不超过26%，2011年录取率仅17%。

    IJCAI每年会评选多个奖项，包含嘉奖人工智能领域杰出年轻科学家的IJCAI Computers and Thought Award、针对在职业生涯中为人工智能领域做出突出贡献和服务的高级科学家的Donald E. Walker Distinguished Service Award、面向进行了连续性高质量研究并取得实质性成果的科学家的IJCAI Award for Research Excellence以及若干最佳论文奖。


3. NLP领域的四大金刚系列:

 - [ACL](https://www.aclweb.org/portal/) 

    ACL: Association of Computational Linguistics

    计算语言学协会是处理自然语言和计算问题的国际专业协会，成立于1962年，最初名为机器翻译和计算语言协会，于1968年更名为计算语言学协会。ACL每年选择不同的地点，一般在夏天举办年度会议，2015年第53届ACL年会在北京举办。2019年7月28日到8月2日第57届ACL会议将于意大利佛罗伦萨举办。

 - [NAACL-HLT](http://naacl.org/) 

    NAACL-HLT: Annual Conference of the North American Chapter of the Association for Computational Linguistics : Human Language Technologies

    ACL有一个欧洲分会（EACL）和一个北美分会（NAACL）。NAACL-HLT即是这个北美分会，一般简称为NAACL，HLT是强调对人类语言技术的专注和重视。虽然作为分会，但NAACL在自然语言处理领域也是当之无愧的顶级会议，每年选择在一个北美城市召开会议。2019年NAACL会议将于6月2日到7日在美国明尼阿波利斯举办。

 - [EMNLP](https://www.aclweb.org) 

    EMNLP: Empirical Methods in Natural Language Processing

    EMNLP由ACL当中对语言数据和经验方法有特殊兴趣的团体主办，始于1996年。2019年EMNLP会议将会于11月3日到7日于香港亚洲世博会举办。关于论文接收等会议信息可以在ACL2019的网站中查询。

 - [COLING](https://www.sheffield.ac.uk/dcs/research/groups/nlp/iccl/index#tab00) 

    COLING: International Conference on Computational Linguistics

    前三个会议都是ACL主办，这个终于不是了，但是为什么名字和全称完全不对应呢？原来这个会议始于1965年，是由ICCL国际计算语言学委员会（International Committee on Computational Linguistics）主办，自从第三届会议在斯德哥尔摩举办之后，这个会议就有了一个简称COLING，是谐音瑞典著名作家 Albert Engström小说中的虚构人物Kolingen。所以COLING一词和其代表的计算语言学国际会议在字面上并不对应。

3. CV 领域的三剑客系列:

 - [ICCV](http://iccv2019.thecvf.com/submission/timeline) 

    ICCV: International Conference on Computer Vision

    国际计算机视觉会议是由IEEE主办，通常持续4到5天，第1天由特定领域的顶尖专家进行指导讲座，接下来是技术会议和海报展示。ICCV会议设立了多个计算机视觉领域重量级的奖项，例如Azriel Rosenfeld Liftetime Achievement Award以计算机科学家和数学家Azriel Rosenfeld命名，每两年颁布一次，奖励在职业生涯中对计算机视觉做出突出贡献的研究者。每年会议中的最佳论文会被授予Marr Prize，是计算机视觉研究者的最高荣誉之一。

 - [CVPR](http://cvpr2019.thecvf.com/) 

    CVPR: Conference on Computer Vision and Pattern Recognition

    CVPR最初由IEEE计算机协会主办，2012年之后由IEEE计算机协会和CVF计算机视觉基金会联合主办，每年都会列出一张明确的主题清单，包含了众多专题研讨会和辅导课。CVPR采用多层级双盲评审流程，一篇论文要先后经过不少于3个评审员、数个区域主席、数个项目主席的评审，论文录取率小于30%，口头报告的录取率小于5%。

    CVPR也设置了数个奖项，包括CVPR Best Paper Award, Longuet-Higgins Prize以及PAMI Young Researcher Award等。

 - [ECCV](https://eccv2020.eu/) 

    ECCV: European Conference on Computer Vision

    欧洲计算机视觉国际会议每两年举办一次，包含辅导课、技术分享和海报展示，收录的论文主要来自于欧洲和美国的顶尖实验室或研究所。

### 论文搜索和分析工具

#### Google scholar

##### 介绍

##### 相关链接

1. [Google scholar](https://scholar.google.co.il/)

##### 效果图

![](https://imgkr2.cn-bj.ufileos.com/4e1ba94f-42cb-48f8-bafb-d9f13fc4c5ee.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=vBM3st9loAP6E4B9%252FHOf58laCRg%253D&Expires=1597155469)


#### Semantic scholar

##### 介绍

##### 相关链接

1. [Semantic scholar](https://www.semanticscholar.org/)

##### 效果图


![](https://imgkr2.cn-bj.ufileos.com/888b761c-8bae-4474-8580-6b0800425fb9.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=cVzWIpK3l8gi1XS%252BoXOJJTWzWzA%253D&Expires=1597155487)

#### 计算机视觉方面的论文

##### 介绍

记录每天整理的计算机视觉/深度学习/机器学习相关方向的论文

##### 相关链接

1. [计算机视觉方面的论文](https://github.com/amusi/daily-paper-computer-vision)

##### 效果图

![](https://imgkr2.cn-bj.ufileos.com/f0dcecaf-9900-482b-a9f9-d4b01c98c4c6.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=DLr4tMInMW4qyUCCRU85ips9jd8%253D&Expires=1597155494)

#### Deep Learning Monitor

##### 介绍

##### 相关链接

1. [Deep Learning Monitor](https://deeplearn.org/)

##### 效果图

![](https://imgkr2.cn-bj.ufileos.com/dab1b8e3-4567-4195-a517-461ae82db312.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=CqY7L7XE932IsYKHIP%252BzxNDJo1I%253D&Expires=1597155520)

## 外文读不懂怎么办？

### arXiv-sanity

#### 介绍

相比于 arXiv 有很大的改进，包括在浏览中显示摘要、评论和非常基本的社交、库功能。这个整合了很多便捷功能的网站，是 Andrej Karpathy 在空闲时开发的。

#### 相关链接

1. [arXiv-sanity 官网](http://arxiv-sanity.com/)

#### 效果图


![](https://imgkr2.cn-bj.ufileos.com/1a90c094-4618-4b0a-9ed7-a1a175d9eee5.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=ZCZOoB%252FFKdpqPzQ1FCShaUemWf0%253D&Expires=1597155538)


### Arxiv Vanity

#### 介绍

可以将来自 arXiv 的论文渲染成响应式网页，从而让人们不用再看 pdf 文档。

#### 相关链接

1. [arXiv-sanity 官网](http://arxiv-sanity.com/)

2. [arXiv-sanity 谷歌插件](https://chrome.google.com/webstore/detail/arxiv-vanity-plugin/jfnlkegibnoaagfdabjkchhocdhnoofk?type=ext&hl=zh-TW)

#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/e4646aa8-1434-4f3e-8cc8-a807061e0f64.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=no1UhvVaEX4WUWw33Cu9gHQMiXc%253D&Expires=1597155553)

### 论文翻译神器 ———— 通天塔

#### 介绍

可以查看一些论文的译文

#### 相关链接

1. [通天塔 官网](https://tongtianta.site/)


#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/be36d6f4-4a21-4ce5-9d9b-e7b2025f2c2c.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=zfVEXrifjXOdVAPtvPLrafQgbh4%253D&Expires=1597155570)

### 论文翻译小助手 ———— 彩云小译

#### 介绍

谷歌插件：可以实现在线翻译外文网页；

官网：可以上传并在线翻译 外文 pdf;

#### 相关链接

1. [彩云小译官网](https://fanyi.caiyunapp.com/#/)

2. [彩云小译谷歌插件](https://chrome.google.com/webstore/detail/lingocloud-web-translatio/jmpepeebcbihafjjadogphmbgiffiajh?hl=zh-CN)


#### 使用方法介绍

##### 官网篇

step 1: 官网界面

![](https://imgkr2.cn-bj.ufileos.com/c5138efa-17ca-459a-b6ca-8fff0bf75e4a.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=xjk2%252BapQ3DCodplB0quVAzc%252BbDU%253D&Expires=1597155586)

step 2: 上传 pdf

![](https://imgkr2.cn-bj.ufileos.com/f99c710d-2ab9-4d2c-a6b0-f27af7a24f0c.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=didFdt9HJ8LSzrHza5wWcqsbRGs%253D&Expires=1597155597)

step 3: 选择翻译模式

![](https://imgkr2.cn-bj.ufileos.com/adf0e6be-d77d-4772-b9ae-aab42a8d1739.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=H%252Fym%252BgwAv7P08Y1I8tdhkr%252BEsTY%253D&Expires=1597155604)

step 4: 翻译ing!!!

![](https://imgkr2.cn-bj.ufileos.com/821a08bf-18bb-4e07-8635-d89da5bd6980.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=SCyJ029x%252B1kTx2w%252FZ5VMx4uFEgM%253D&Expires=1597155610)

step 5: 翻译效果

见证奇迹的时刻！！！
见证奇迹的时刻！！！
见证奇迹的时刻！！！

![](https://imgkr2.cn-bj.ufileos.com/f1eaf893-1310-4814-8c67-fd25d609042f.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=FSfvmuZSlFmuhbPLTutWIJtaOw8%253D&Expires=1597155619)

意不意外？激不激动？兴不兴奋？？？

##### 谷歌插件篇

step 1：下载插件，并安装插件

![](https://imgkr2.cn-bj.ufileos.com/9fa55a71-5270-436f-9ade-247bc43d5ff1.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=iIyXJhbe%252F%252FoWdskp86geWkO0wEs%253D&Expires=1597155632)

step 2：使用插件

![](https://imgkr2.cn-bj.ufileos.com/294fe7e2-c0ca-4f9f-8de8-eaea7da47fb4.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=GoJf%252FyK3kPpMXrPY2Ha1%252FAwNLkQ%253D&Expires=1597155639)

step 3：翻译效果

![](https://imgkr2.cn-bj.ufileos.com/f9832170-6286-4b2d-be7c-46c0cabe432c.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=OzkW%252Bm%252BqvP0TPZndDUadC5HNzD8%253D&Expires=1597155650)

## 外文没 code 怎么办？

### papers with code

#### 介绍

各个领域的最新论文及对应的 code。

#### 相关链接

1. [papers with code 官网](https://paperswithcode.com/sota)

#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/4141cff3-0889-45ed-9b15-b734de70318b.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=HqjR6gSvTg3mUsbs17ZwXky89uU%253D&Expires=1597155658)

![](https://imgkr2.cn-bj.ufileos.com/68126b0e-8c5d-4645-949c-385e802ed7b3.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=oLCdzPixIRRPuhUaD3t%252Fq0jDaqE%253D&Expires=1597155665)

### OpenGitHub 新项目快报

#### 介绍

可以查看 GitHub 最新的项目资源。

#### 相关链接

1. [OpenGitHub 新项目快报 官网](https://www.open-open.com/github/view/github2019-10-23.html)

#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/3ac96dde-f1ec-4851-9968-5ad415273b07.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=w%252FKHYTjwhbQmnwOPnVYv%252FzLGjuo%253D&Expires=1597155678)

### New Standard In AI

#### 介绍

可能是地表最大的AI算法市场 (AI MARKET)

#### 相关链接

1. [New Standard In AI 官网](http://manaai.cn/)

#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/96b1ffe5-a814-4a11-9d69-e016d2060226.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=EPtqb42JDts6qhOBzGzYt5%252BosAA%253D&Expires=1597155808)

### Github pwc

#### 介绍

Papers with code. Sorted by stars. Updated weekly.

#### 相关链接

1. [Github pwc 官网](http://manaai.cn/)

#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/bec024bf-4748-40e9-91bb-b1e79b83e6ee.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=xPKBEi%252FCC4XCz%252FEDLx21aChhl6g%253D&Expires=1597155819)

## 外文写起来麻烦怎么办？

### Overleaf

#### 介绍

支持多人协作的在线 LaTeX 编辑器，好比用谷歌文档写论文，很好实现

#### 相关链接

1. [Overleaf 官网](https://www.overleaf.com/)

#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/4097e34e-b2f6-4712-813b-9d2fd9862350.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=Hjx4ve0UEH3eXpJMjGAH4z%252BRdQA%253D&Expires=1597155855)


### Authorea

#### 介绍

一种支持多人协作在线撰写论文的方法，旨在减少 LaTeX 的使用，支持现代 WYSIWYG 编辑器。支持内联代码和数据，促进可复现性，支持内联公共评论和其它合理功能

#### 相关链接

1. [Authorea 官网](https://www.authorea.com/)

#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/835dc0a9-4d70-4d9f-97ea-83f93e147123.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=FczYO47WiLpzpJVqQdo3Qw0KeNk%253D&Expires=1597155864)

### Code ocean

#### 介绍

基于云计算的再现性平台。我的理解是你将自己的研究作为 Jupyter 环境代码上传，然后在线运行，并复现作者曾取得的相同图表/输出。

#### 相关链接

1. [Code ocean 官网](https://codeocean.com/)

#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/d4737838-6a9a-43f1-8a8d-f4cbd68965f4.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=CWOBN4itBBsmIFzXnp8H6oqjdrw%253D&Expires=1597155872)

## 搞科研没人交流怎么办？

### Shortscience

#### 介绍

这是一个能共享论文概述的平台，目前有超过 1000 篇论文概述，并仍在持续增长

#### 相关链接

1. [Shortscience 官网](http://www.shortscience.org/)

#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/2fe6d87f-5a3e-47e8-820a-23459a113eb4.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=1axkCK5KaW9S5Xhp66jhfAmeGYQ%253D&Expires=1597155878)

### OpenReview

#### 介绍

这是一个能提供公开论文评审过程的平台，所有提交的论文会公开作者姓名等信息，同时接受同行的评价及提问，可以匿名或实名地对论文进行评价。公开评审结束后，论文作者也能够调整和修改论文。Openreview 目前仅对特定学术会议提供评审功能，例如 ICLR，并且由于受到广泛质疑，ICLR 在 Openreview 上的评审也被改成了双盲评审。除了官方评审之外，近期很多论文的评论区也能看到读者和作者之间的积极交流


#### 相关链接

1. [OpenReview 官网](https://openreview.net/)

#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/893b779a-ee49-49e5-b59f-ec3f156fa2a1.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=AFBec%252B7MZRxMPiVHmgkWZ4f5fhs%253D&Expires=1597155889)

### Scirate

#### 介绍

能看到热度较高的 arXiv 论文，并按学科分门别类，还能浏览相关论文的评论。但其热度排序基于该网站内的点赞数，而这个网站的活跃度并不高

#### 相关链接

1. [Scirate 官网](https://scirate.com/)

#### 效果图

![](https://imgkr2.cn-bj.ufileos.com/ea583305-9bed-48dd-9b98-55dd321bba40.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=buLCsCtbj8oZCK79tqIm5%252B9oDZY%253D&Expires=1597155896)

## 参考网站

1. [盘点AI国际顶级会议](https://zhuanlan.zhihu.com/p/51749414)

2. [看论文的好工具](https://www.jianshu.com/p/d61aa4c02ef6)

3. [死磕论文前，不如先找齐一套好用的工具](https://zhuanlan.zhihu.com/p/49856162)