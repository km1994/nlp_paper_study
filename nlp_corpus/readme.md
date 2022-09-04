# 【关于 NLP 语料】那些你不知道的事

## 八、对话语料

### 8.1 对话语料介绍

语料名称 | 语料数量 | 语料来源说明 | 语料特点 | 语料样例 | 是否已分词
---|---|---|---|---|---
chatterbot | 560 | 开源项目 | 按类型分类，质量较高  | Q:你会开心的 A:幸福不是真正的可预测的情绪。 | 否
douban（豆瓣多轮） | 352W | 来自北航和微软的paper, 开源项目 | 噪音相对较少，原本是多轮（平均7.6轮）  | Q:烟台 十一 哪 好玩 A:哪 都 好玩 · · · · | 是
ptt（PTT八卦语料） | 77W（v1版本42W） | 开源项目，台湾PTT论坛八卦版 | 繁体，语料较生活化，有噪音  | Q:为什么乡民总是欺负国高中生呢QQ	A:如果以为选好科系就会变成比尔盖兹那不如退学吧  | 否
qingyun（青云语料） | 10W | 某聊天机器人交流群 | 相对不错，生活化  | Q:看来你很爱钱 	 A:噢是吗？那么你也差不多了 | 否
subtitle（电视剧对白语料） | 274W | 开源项目，来自爬取的电影和美剧的字幕 | 有一些噪音，对白不一定是严谨的对话，原本是多轮（平均5.3轮）  | Q:京戏里头的人都是不自由的	A:他们让人拿笼子给套起来了了 | 否
tieba（贴吧论坛回帖语料） | 232W | 偶然找到的 | 多轮，有噪音  | Q:前排，鲁迷们都起床了吧	A:标题说助攻，但是看了那球，真是活生生的讽刺了 | 否
weibo（微博语料） | 443W | 来自华为的paper | 仍有一些噪音  | Q:北京的小纯洁们，周日见。#硬汉摆拍清纯照# A:嗷嗷大湿的左手在干嘛，看着小纯洁撸么。 | 否
xiaohuangji（小黄鸡语料） | 45W | 原人人网项目语料 | 有一些不雅对话，少量噪音 | Q:你谈过恋爱么	A:谈过，哎，别提了，伤心..。 | 否

语料名称 | 语料原始URL（即出处，尊重原始版权） 
---|---
chatterbot | https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/chinese
douban（豆瓣多轮） | https://github.com/MarkWuNLP/MultiTurnResponseSelection 
ptt（PTT八卦语料）| https://github.com/zake7749/Gossiping-Chinese-Corpus 
qingyun（青云语料） | 无 
subtitle（电视剧对白语料） | https://github.com/fateleak/dgk_lost_conv 
tieba（贴吧论坛回帖语料）  | https://pan.baidu.com/s/1mUknfwy1nhSM7XzH8xi7gQ 密码:i4si 
weibo（微博语料）  | 61.93.89.94/Noah_NRM_Data/ 
xiaohuangji（小黄鸡语料） | https://github.com/candlewill/Dialog_Corpus 

### 8.2 对话语料下载

1. 百度网盘链接: https://pan.baidu.com/s/17rIIkhQN2ZNULOGvfE2OWQ  密码: iani

> 因为一些原因，百度网盘的链接不包含PTT语料，请自行下载放在相应目录

2. Google Drive  https://drive.google.com/file/d/1So-m83NdUHexfjJ912rQ4GItdLvnmJMD/view?usp=sharing


## 七、文本匹配数据

### 7.1 afqmc数据集

总样本数为：38650，其中，匹配样本个数为：11911，不匹配样本个数为：26739

### 7.2 ccks2018_task3 数据集

总样本数为：100000，其中，匹配样本个数为：50000，不匹配样本个数为：50000

### 7.3 chip2019数据集

总样本数为：20000，其中，匹配样本个数为：10000，不匹配样本个数为：10000

### 7.4 COVID-19数据集

总样本数为：10749，其中，匹配样本个数为：4301，不匹配样本个数为：6448

### 7.5 diac2019数据集

总样本数为：100298，其中，匹配样本个数为：38446，不匹配样本个数为：61852

### 7.6 gaiic2021_task3数据集

总样本数为：177173，其中，匹配样本个数为：54805，不匹配样本个数为：122368

### 7.7 lcqmc数据集

总样本数为：260068，其中，匹配样本个数为：149226，不匹配样本个数为：110842

### 7.8 pawsx数据集

总样本数为：53401，其中，匹配样本个数为：23576，不匹配样本个数为：29825

### 7.9 ths2021数据集

总样本数为：41756，其中，匹配样本个数为：10478，不匹配样本个数为：31278

### 7.10 xf2021数据集

总样本数为：5000，其中，匹配样本个数为：2892，不匹配样本个数为：2108

### 7.11 sohu_2021数据集

总样本数为：69578，其中，匹配样本个数为：18714，不匹配样本个数为：50864

### 7.12 cmnli数据集

总样本数为：404024，其中，匹配样本个数为：134889，不匹配样本个数为：269135

### 7.13 csnli数据集

总样本数为：564339，其中，匹配样本个数为：188518，不匹配样本个数为：375821

### 7.14 ocnli数据集

总样本数为：53387，其中，匹配样本个数为：17726，不匹配样本个数为：35661

### 7.15 cstsb数据集

总样本数为：4473，其中，匹配样本个数为：401，不匹配样本个数为：4072

### 7.16 pku数据集

总样本数为：509832，其中，匹配样本个数为：509832，不匹配样本个数为：0


### 参考

- [中文文本匹配数据集整理](https://github.com/liucongg/NLPDataSet)


## 六、文本摘要数据

### 6.4 nlpcc 自动摘要英文语料库

```s
{
  "version": "0.0.1",
  "data":
  [
      {
          "title": "知情人透露章子怡怀孕后,父母很高兴。章母已开始悉心照料。据悉,预产期大概是12月底",
          "content": "四海网讯,近日,有媒体报道称:章子怡真怀孕了!报道还援引知情人士消息称,“章子怡怀孕大概四五个月,预产期是年底前后,现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息,23日晚8时30分,华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士,这位人士向华西都市报记者证实说:“子怡这次确实怀孕了。她已经36岁了,也该怀孕了。章子怡怀上汪峰的孩子后,子怡的父母亲十分高兴。子怡的母亲,已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时,华西都市报记者为了求证章子怡怀孕消息,又电话联系章子怡的亲哥哥章子男,但电话通了,一直没有人接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来,就被传N遍了!不过,时间跨入2015年,事情却发生着微妙的变化。2015年3月21日,章子怡担任制片人的电影《从天儿降》开机,在开机发布会上几张合影,让网友又燃起了好奇心:“章子怡真的怀孕了吗?”但后据证实,章子怡的“大肚照”只是影片宣传的噱头。过了四个月的7月22日,《太平轮》新一轮宣传,章子怡又被发现状态不佳,不时深呼吸,不自觉想捂住肚子,又觉得不妥。然后在8月的一天,章子怡和朋友吃饭,在酒店门口被风行工作室拍到了,疑似有孕在身!今年7月11日,汪峰本来在上海要举行演唱会,后来因为台风“灿鸿”取消了。而消息人士称,汪峰原来打算在演唱会上当着章子怡的面宣布重大消息,而且章子怡已经赴上海准备参加演唱会了,怎知遇到台风,只好延期,相信9月26日的演唱会应该还会有惊喜大白天下吧。"
      },...
  ]
}
```

### 6.3 SQuAD 自动摘要英文语料库

```s
  {"data": [{"title": "Super_Bowl_50", "paragraphs": [{"context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.", "qas": [{"answers": [{"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}], "question": "Which NFL team represented the AFC at Super Bowl 50?", "id": "56be4db0acb8001400a502ec"}, {"answers": [{"answer_start": 249, "text": "Carolina Panthers"}, {"answer_start": 249, "text": "Carolina Panthers"}, {"answer_start": 249, "text": "Carolina Panthers"}], "question": "Which NFL team represented the NFC at Super Bowl 50?", "id": "56be4db0acb8001400a502ed"}, {"answers": [{"answer_start": 403, "text": "Santa Clara, California"}, {"answer_start": 355, "text": "Levi's Stadium"}, {"answer_start": 355, "text": "Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."}], "question": "Where did Super Bowl 50 take place?", "id": "56be4db0acb8001400a502ee"}, {"answers": [{"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}], "question": "Which NFL team won Super Bowl 50?", "id": "56be4db0acb8001400a502ef"}, {"answers": [{"answer_start": 488, "text": "gold"}, {"answer_start": 488, "text": "gold"}, {"answer_start": 521, "text": "gold"}], "question": "What color was used to emphasize the 50th anniversary of the Super Bowl?", "id": "56be4db0acb8001400a502f0"}, {"answers": [{"answer_start": 487, "text": "\"golden anniversary\""}, {"answer_start": 521, "text": "gold-themed"}, {"answer_start": 487, "text": "\"golden anniversary"}], "question": "What was the theme of Super Bowl 50?", "id": "56be8e613aeaaa14008c90d1"}, {"answers": [{"answer_start": 334, "text": "February 7, 2016"}, {"answer_start": 334, "text": "February 7"}, {"answer_start": 334, "text": "February 7, 2016"}], "question": "What day was the game played on?", "id": "56be8e613aeaaa14008c90d2"}, {"answers": [{"answer_start": 133, "text": "American Football Conference"}, {"answer_start": 133, "text": "American Football Conference"}, {"answer_start": 133, "text": "American Football Conference"}], "question": "What is the AFC short for?", "id": "56be8e613aeaaa14008c90d3"}, {"answers": [{"answer_start": 487, "text": "\"golden anniversary\""}, {"answer_start": 521, "text": "gold-themed"}, {"answer_start": 521, "text": "gold"}], "question": "What was the theme of Super Bowl 50?", "id": "56bea9923aeaaa14008c91b9"}, {"answers": [{"answer_start": 133, "text": "American Football Conference"}, {"answer_start": 133, "text": "American Football Conference"}, {"answer_start": 133, "text": "American Football Conference"}], "question": "What does AFC stand for?", "id": "56bea9923aeaaa14008c91ba"}, {"answers": [{"answer_start": 334, "text": "February 7, 2016"}, {"answer_start": 334, "text": "February 7"}, {"answer_start": 334, "text": "February 7, 2016"}], "question": "What day was the Super Bowl played on?", "id": "56bea9923aeaaa14008c91bb"}, {"answers": [{"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}], "question": "Who won Super Bowl 50?", "id": "56beace93aeaaa14008c91df"}, {"answers": [{"answer_start": 355, "text": "Levi's Stadium"}, {"answer_start": 355, "text": "Levi's Stadium"}, {"answer_start": 355, "text": "Levi's Stadium in the San Francisco Bay Area at Santa Clara"}], "question": "What venue did Super Bowl 50 take place in?", "id": "56beace93aeaaa14008c91e0"}, {"answers": [{"answer_start": 403, "text": "Santa Clara"}, {"answer_start": 403, "text": "Santa Clara"}, {"answer_start": 403, "text": "Santa Clara"}], "question": "What city did Super Bowl 50 take place in?", "id": "56beace93aeaaa14008c91e1"}, {"answers": [{"answer_start": 693, "text": "Super Bowl L"}, {"answer_start": 704, "text": "L"}, {"answer_start": 693, "text": "Super Bowl L"}], "question": "If Roman numerals were used, what would Super Bowl 50 have been called?", "id": "56beace93aeaaa14008c91e2"}, {"answers": [{"answer_start": 116, "text": "2015"}, {"answer_start": 112, "text": "the 2015 season"}, {"answer_start": 116, "text": "2015"}], "question": "Super Bowl 50 decided the NFL champion for what season?", "id": "56beace93aeaaa14008c91e3"}, {"answers": [{"answer_start": 116, "text": "2015"}, {"answer_start": 346, "text": "2016"}, {"answer_start": 116, "text": "2015"}], "question": "What year did the Denver Broncos secure a Super Bowl title for the third time?", "id": "56bf10f43aeaaa14008c94fd"}, {"answers": [{"answer_start": 403, "text": "Santa Clara"}, {"answer_start": 403, "text": "Santa Clara"}, {"answer_start": 403, "text": "Santa Clara"}], "question": "What city did Super Bowl 50 take place in?", "id": "56bf10f43aeaaa14008c94fe"}, {"answers": [{"answer_start": 355, "text": "Levi's Stadium"}, {"answer_start": 355, "text": "Levi's Stadium"}, {"answer_start": 355, "text": "Levi's Stadium"}], "question": "What stadium did Super Bowl 50 take place in?", "id": "56bf10f43aeaaa14008c94ff"}, {"answers": [{"answer_start": 267, "text": "24\u201310"}, {"answer_start": 267, "text": "24\u201310"}, {"answer_start": 267, "text": "24\u201310"}], "question": "What was the final score of Super Bowl 50? ", "id": "56bf10f43aeaaa14008c9500"}, {"answers": [{"answer_start": 334, "text": "February 7, 2016"}, {"answer_start": 334, "text": "February 7, 2016"}, {"answer_start": 334, "text": "February 7, 2016"}], "question": "What month, day and year did Super Bowl 50 take place? ", "id": "56bf10f43aeaaa14008c9501"}, {"answers": [{"answer_start": 116, "text": "2015"}, {"answer_start": 346, "text": "2016"}, {"answer_start": 346, "text": "2016"}], "question": "What year was Super Bowl 50?", "id": "56d20362e7d4791d009025e8"}, {"answers": [{"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}], "question": "What team was the AFC champion?", "id": "56d20362e7d4791d009025e9"}, {"answers": [{"answer_start": 249, "text": "Carolina Panthers"}, {"answer_start": 249, "text": "Carolina Panthers"}, {"answer_start": 249, "text": "Carolina Panthers"}], "question": "What team was the NFC champion?", "id": "56d20362e7d4791d009025ea"}, {"answers": [{"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}], "question": "Who won Super Bowl 50?", "id": "56d20362e7d4791d009025eb"}, {"answers": [{"answer_start": 116, "text": "2015"}, {"answer_start": 112, "text": "the 2015 season"}, {"answer_start": 116, "text": "2015"}], "question": "Super Bowl 50 determined the NFL champion for what season?", "id": "56d600e31c85041400946eae"}, {"answers": [{"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}], "question": "Which team won Super Bowl 50.", "id": "56d600e31c85041400946eb0"}, {"answers": [{"answer_start": 403, "text": "Santa Clara, California."}, {"answer_start": 355, "text": "Levi's Stadium"}, {"answer_start": 355, "text": "Levi's Stadium"}], "question": "Where was Super Bowl 50 held?", "id": "56d600e31c85041400946eb1"}, {"answers": [{"answer_start": 0, "text": "Super Bowl"}, {"answer_start": 0, "text": "Super Bowl"}, {"answer_start": 0, "text": "Super Bowl"}], "question": "The name of the NFL championship game is?", "id": "56d9895ddc89441400fdb50e"}, {"answers": [{"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}, {"answer_start": 177, "text": "Denver Broncos"}], "question": "What 2015 NFL team one the AFC playoff?", "id": "56d9895ddc89441400fdb510"}]},
  ...
}

```


### 6.2 lcsts 生成式自自动摘要中文语料库

- article.txt

```s
  新华社受权于18日全文播发修改后的《中华人民共和国立法法》，修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。
  一辆小轿车，一名女司机，竟造成9死24伤。日前，深圳市交警局对事故进行通报：从目前证据看，事故系司机超速行驶且操作不当导致。目前24名伤员已有6名治愈出院，其余正接受治疗，预计事故赔偿费或超一千万元。
  1月18日，习近平总书记对政法工作作出重要指示：2014年，政法战线各项工作特别是改革工作取得新成效。新形势下，希望全国政法机关主动适应新形势，为公正司法和提高执法司法公信力提供有力制度保障。
  针对央视3·15晚会曝光的电信行业乱象，工信部在公告中表示，将严查央视3·15晚会曝光通信违规违法行为。工信部称，已约谈三大运营商有关负责人，并连夜责成三大运营商和所在省通信管理局进行调查，依法依规严肃处理。
  ...
```

- summary.txt

```s
  修改后的立法法全文公布
  深圳机场9死24伤续：司机全责赔偿或超千万
  孟建柱：主动适应形势新变化提高政法机关服务大局的能力
  工信部约谈三大运营商严查通信违规
  ...
```


### 6.1 教育培训行业抽象式自动摘要中文语料库

自动文摘分为两种：

1. 抽取式

2. 抽象式

语料库收集了教育培训行业主流垂直媒体的历史文章（截止到2018年6月5日）大约24500条数据集。主要是为训练抽象式模型而整理，每条数据有summary(摘要)和text(正文)，两个字段，Summary字段均为作者标注。

压缩包大约 60 MB，解压后大约 150 MB。

![sample](img/sample.png)

格式如下：

summary{{...}}

text{{...}}


## 五、事件抽取数据集

- [数据资源：事件图谱构建中常用事件抽取、因果事件关系数据集的总结与思考](https://mp.weixin.qq.com/s/feobmsEHINwM-UZbHVzq2w)

## 四、文本分类

### 4.8 带情感标注 amazon  【 yf_amazon 】

- 介绍：豆瓣电影 情感/观点/评论 倾向性分析
- 时间：
- 字段说明

| 字段 | 说明 |
| ---- | ---- |
| productId | 产品 id (从 0 开始，连续编号) |
| name | 产品名称 |
| catIds | 类别 id（从 0 开始，连续编号，从左到右依次表示一级类目、二级类目、三级类目） |

- 数据集：52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据
- 地址： https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/yf_amazon/intro.ipynb
- 数据格式

```s
	productId	name	catIds
331420	331420	欧意金狐狸 女式 皮手套 QT602	802,143,996
130945	130945	YESO TOT 中性 单肩包/斜挎包 均码 9411	1111,864,781
179886	179886	李斯特论柏辽兹与舒曼	832,552,337
    ...
```

### 4.7 带情感标注 餐馆名称  【 yf_dianping 】

- 介绍：餐馆名称 情感/观点/评论 倾向性分析
- 时间：
- 字段说明

| 字段 | 说明 |
| ---- | ---- |
| restId | 餐馆 id (从 0 开始，连续编号) |
| name | 餐馆名称 |

- 数据集：24 万家餐馆，54 万用户，440 万条评论/评分数据
- 地址： https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/yf_dianping/intro.ipynb
- 数据格式

```s
  restId	name
  210902	210902	NaN
  124832	124832	NaN
  26766	26766	香锅制造(新苏天地店)
  91754	91754	NaN
    ...
```

### 4.6 带情感标注 豆瓣电影  【 dmsc_v2 】

- 介绍：豆瓣电影 情感/观点/评论 倾向性分析
- 时间：
- 字段说明

| 字段 | 说明 |
| ---- | ---- |
| movieId | 电影 id (从 0 开始，连续编号) |
| title | 英文名称 |
| title_cn | 中文名称 |

- 数据集：28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据
- 地址： https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/dmsc_v2/intro.ipynb
- 数据格式

```s
  movieId	title	title_cn
  0	0	Avengers Age of Ultron	复仇者联盟2
  1	1	Big Fish and Begonia	大鱼海棠
  2	2	Captain America Civil War	美国队长3
  3	3	Chinese Zodiac	十二生肖
    ...
```

### 4.5 带情感标注 新浪微博  【simplifyweibo_4_moods 】

- 介绍：情感/观点/评论 倾向性分析
- 时间：
- 字段说明

| 字段 | 说明 |
| ---- | ---- |
| label | 0 喜悦，1 愤怒，2 厌恶，3 低落 |
| review | 微博内容 |

- 数据集：36 万多条，带情感标注 新浪微博，包含 4 种情感，其中喜悦约 20 万条，愤怒、厌恶、低落各约 5 万条
- 地址： https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb
- 数据格式

```s
  	  	label	review
  62050	0	太过分了@Rexzhenghao //@Janie_Zhang:招行最近负面新闻越来越多呀...
  68263	0	希望你?得好?我本＂?肥血?史＂[晕][哈哈]@Pete三姑父
  81472	0	有点想参加????[偷?]想安排下时间再决定[抓狂]//@黑晶晶crystal: @细腿大羽...
    ...
```

### 4.4 带情感标注 新浪微博  【weibo_senti_100k 】

- 介绍：带情感标注 新浪微博
- 时间：
- 字段说明

| 字段 | 说明 |
| ---- | ---- |
| label | 1 表示正向评论，0 表示负向评论 |
| review | 微博内容 |

- 数据集：10 万多条，带情感标注 新浪微博，正负向评论约各 5 万条
- 地址： https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb
- 数据格式

```s
  	  	label	review
  62050	0	太过分了@Rexzhenghao //@Janie_Zhang:招行最近负面新闻越来越多呀...
  68263	0	希望你?得好?我本＂?肥血?史＂[晕][哈哈]@Pete三姑父
  81472	0	有点想参加????[偷?]想安排下时间再决定[抓狂]//@黑晶晶crystal: @细腿大羽...
    ...
```

### 4.3 情感/观点/评论 倾向性分析  【online_shopping_10_cats】

- 介绍：情感/观点/评论 倾向性分析
- 时间：
- 字段说明

| 字段 | 说明 |
| ---- | ---- |
| cat | 类别：包括 书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店 |
| label | 1 表示正向评论，0 表示负向评论 |
| review | 评论内容 |

- 数据集：10 个类别（书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店），共 6 万多条评论数据，正、负向评论各约 3 万条
- 地址： https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb
- 数据格式

```s
  	  	cat	label	review
  11194	平板	0	什么玩意。刚用一天，就充不上电，开不开机，返厂老麻烦，
  17794	水果	1	买了几次了，价格实惠，口感不错，保鲜好！
  29529	洗发水	1	挺值得购买的，有包装买回去送家人，毛巾质量不错。小块的可以拿来当擦手帕。
    ...
```

### 4.2 外卖平台收集的用户评价 【waimai_10k】

- 介绍：外卖平台收集的用户评价
- 时间：
- 字段说明

| 字段 | 说明 |
| ---- | ---- |
| label | 1 表示正向评论，0 表示负向评论 |
| review | 评论内容 |

- 数据集：正向 4000 条，负向 约 8000 条
- 地址： https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/waimai_10k
- 数据格式

```s
  label,review
  1,很快，好吃，味道足，量大
  1,没有送水没有送水没有送水
  1,非常快，态度好。
  1,方便，快捷，味道可口，快递给力
  1,菜味道很棒！送餐很及时！
  1,今天师傅是不是手抖了，微辣格外辣！
    ...
```

### 4.1 酒店评论数据 【ChnSentiCorp_htl_all】

- 介绍：酒店评论数据
- 时间：
- 字段说明

| 字段 | 说明 |
| ---- | ---- |
| label | 1 表示正向评论，0 表示负向评论 |
| review | 评论内容 |

- 数据集：7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论
- 地址： https://github.com/SophonPlus/ChineseNlpCorpus/raw/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv
- 数据格式

```s
  	  label	review
  5612	0	房间小得无法想象,建议个子大的不要选择,一般的睡觉脚也伸不直.房间不超过10平方,彩电是14...
  7321	0	我们一家人带孩子去过“五.一”，在协程网上挑了半天才选中的酒店，但看来还是错了。1.酒店除了...
  3870	1	周六到西山去采橘子,路过这家酒店的时候就觉得应该不错的,采好橘子回来天也晚了,就临时决定住在...
    ...
```

## 三、问答数据

### 3.2 保险行业问答数据

- 介绍：8000 多条保险行业问答数据
- 时间：
- 字段说明

| 字段 | 说明 |
| ---- | ---- |
| title | 问题的标题 |
| question | 问题内容（可为空） |
| reply| 回复内容 |
| is_best| 是否为页面上显示的最佳回答 |

- 数据集：8000 多条保险行业问答数据
- 地址： 百度云盘 全部文件>数据集>问答语料>保险行业问答数据
- 数据格式

```s
  title	question	reply	is_best
  6733	五险两金和五险一金有什么区别	单位招聘，独立待遇中有一项是五险两金。有些单位是五险一金，还有些五险两金。然而我刚毕业小白，...	五险一金是指：医疗保险，生育保险，工伤保险，失业保险和养老保险，还有住房公积金。五险两金指的...	0
  7580	户口不在本地如何办医疗保险	户口不在本地如何办医疗保险	户口不在本地可以办理医保，通常都是以单位名义进行办理。医疗保险分两种办理方式，一种是单位办理...	1
  6310	酒精含量百分之二十八保险公司理赔吗？	NaN	不会赔	0
  5843	我买的二手车，车险都没过户，怎么交保险	NaN	要看保险合同了，有的是指定被保险人的，如果你出了险，保险公司是不理赔的。建议尽快去过户，或者...	0
    ...
```

### 3.1 电信问答数据

- 介绍：电信问答数据
- 时间：
- 字段说明

| 字段 | 说明 |
| ---- | ---- |
| title | 标题 |
| question | 问题（可为空） |
| reply| 每个问题的内容 |
| is_best| 是否是最佳答案 |

- 数据集：15.6 万条电信问答数据
- 地址： 百度云盘 全部文件>数据集>问答语料>电信问答数据
- 数据格式

```s
  title	question	reply	is_best
  129754	红米no##4x	NaN	可以，	0
  15843	为什么不能同时用两个电信卡	NaN	您好不可以的，目前推出的手机都是不能同时支持两张电信手机卡的，即使是全网通手机也只能在其中的...	1
  23985	电信181、177、133哪个号段好？	NaN	133的	0
  72065	华*荣耀7x和魅蓝note6哪个好	NaN	荣耀畅玩7X很不错，性价比很高，以下是手机的配置：1、外观方面：荣耀畅玩7X采用5.93英寸...	1
  11843	p8青春版电信版多少钱	NaN	您好，这款手机价格参考如下	1
  3280	华为di####00叫什么	华为di####00叫什么	DI####00是华为畅享6S全网通版。华为畅享6S性价比高,是一款很不错的手机。电信新出流...	1
    ...
```

### 3.6 金融行业问答数据

### 3.5 投资行业问答数据

### 3.4 联通问答数据

### 3.3 农业银行问答数据

## 一、命名实体识别

### 1.1 MSRA-NER实体数据集

- 介绍：由微软亚洲研究院标注的新闻领域的实体识别数据集，也是SIGNAN backoff 2006的实体识别任务的数据集之一。
- 时间：2016
- 实体类型：LOC(地名), ORG(机构名), PER(人名)
- 数据集：训练集46364个句子，验证集4365个句子
- 地址： https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/MSRA
- 数据格式

```s
    {"text": "当希望工程救助的百万儿童成长起来，科教兴国蔚然成风时，今天有收藏价值的书你没买，明日就叫你悔不当初！", "entity_list": []}
    {"text": "藏书本来就是所有传统收藏门类中的第一大户，只是我们结束温饱的时间太短而已。", "entity_list": []}
    {"text": "因有关日寇在京掠夺文物详情，藏界较为重视，也是我们收藏北京史料中的要件之一。", "entity_list": [{"entity_index": {"begin": 3, "end": 4}, "entity_type": "LOC", "entity": "日"}, {"entity_index": {"begin": 6, "end": 7}, "entity_type": "LOC", "entity": "京"}, {"entity_index": {"begin": 27, "end": 29}, "entity_type": "LOC", "entity": "北京"}]}
    ...
```

### 1.2 人民日报实体数据集

- 介绍：以1998年人民日报语料为对象，由北京大学计算语言学研究所和富士通研究开发中心有限公司共同制作的标注语料库。
- 实体类型：LOC(地名), ORG(机构名), PER(人名)
- 数据集：19359条数据集
- 地址： https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/people_daily
- 数据格式

```s
    {"text": "迈向充满希望的新世纪——一九九八年新年讲话(附图片1张)", "entity_list": [{"entity_index": {"begin": 12, "end": 19}, "entity_type": "DATE", "entity": "一九九八年新年"}]}
    {"text": "中共中央总书记、国家主席江泽民", "entity_list": [{"entity_index": {"begin": 0, "end": 4}, "entity_type": "ORG", "entity": "中共中央"}, {"entity_index": {"begin": 12, "end": 15}, "entity_type": "PERSON", "entity": "江泽民"}]}
    ...
```

### 1.3 新浪微博实体数据集

- 介绍：根据新浪微博2013年11月至2014年12月间历史数据筛选过滤生成，包含1890条微博消息，基于LDC2014的DEFT ERE的标注标准进行标注。
- 时间：2014
- 实体类型：地名、人名、机构名、行政区名，并且每个类别可细分为特指（NAM，如“张三”标签为“PER.NAM”）和泛指（NOM，如“男人”标签为“PER.NOM”）。
- 数据集：包括1890条微博消息，发布于2015年。包括1350条训练集、270条验证集、270条测试集。
- 地址：  https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/weibo
- 数据格式

```s
    {"text": "科技全方位资讯智能，快捷的汽车生活需要有三屏一云爱你", "entity_list": []}
    {"text": "对，输给一个女人，的成绩。失望", "entity_list": []}
    {"text": "今天下午起来看到外面的太阳。。。。我第一反应竟然是强烈的想回家泪想我们一起在嘉鱼个时候了。。。。有好多好多的话想对你说李巾凡想要瘦瘦瘦成李帆我是想切开云朵的心", "entity_list": [{"entity_index": {"begin": 38, "end": 39}, "entity_type": "LOC", "entity": "嘉"}, {"entity_index": {"begin": 59, "end": 62}, "entity_type": "PER", "entity": "李巾凡"}, {"entity_index": {"begin": 68, "end": 70}, "entity_type": "PER", "entity": "李帆"}]}
    ...
```

### 1.4 CLUENER细粒度实体数据集

- 介绍：根据清华大学开源的文本分类数据集THUCNEWS，进行筛选过滤、实体标注生成，原数据来源于Sina News RSS。
- 时间：2020
- 实体类型：组织(organization)、人名(name)、地址(address)、公司(company)、政府(government)、书籍(book)、游戏(game)、电影(movie)、职位(position)、景点(scene)等10个实体类别，且实体类别分布较为均衡。
- 数据集：训练集10748个句子，验证集1343个句子
- 地址：  https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/cluener_public
- 数据格式

```s
    {"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", "entity_list": [{"entity_type": "name", "entity": "叶老桂", "entity_index": {"begin": 9, "end": 12}}, {"entity_type": "company", "entity": "浙商银行", "entity_index": {"begin": 0, "end": 4}}]}
    {"text": "生生不息CSOL生化狂潮让你填弹狂扫", "entity_list": [{"entity_type": "game", "entity": "CSOL", "entity_index": {"begin": 4, "end": 8}}]}
    ...
```

### 1.5 Yidu-S4K医疗命名实体识别数据集

- 介绍：源自CCKS2019评测任务一，即“面向中文电子病历的命名实体识别”的数据集。
- 时间：2019
- 实体类型：实验室检验、影像检查、手术、疾病和诊断、药物、解剖部位共6类实体类型。
- 数据集：1000条训练集、379条测试集
- 地址：  https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/yidu-s4k
- 数据格式

```s
    {"text": "，患者3月前因“直肠癌”于在我院于全麻上行直肠癌根治术（DIXON术），手术过程顺利，术后给予抗感染及营养支持治疗，患者恢复好，切口愈合良好。，术后病理示：直肠腺癌（中低度分化），浸润溃疡型，面积3.5*2CM，侵达外膜。双端切线另送“近端”、“远端”及环周底部切除面未查见癌。肠壁一站（10个）、中间组（8个）淋巴结未查见癌。，免疫组化染色示：ERCC1弥漫（+）、TS少部分弱（+）、SYN（-）、CGA（-）。术后查无化疗禁忌后给予3周期化疗，，方案为：奥沙利铂150MG D1，亚叶酸钙0.3G+替加氟1.0G D2-D6，同时给与升白细胞、护肝、止吐、免疫增强治疗，患者副反应轻。院外期间患者一般情况好，无恶心，无腹痛腹胀胀不适，无现患者为行复查及化疗再次来院就诊，门诊以“直肠癌术后”收入院。   近期患者精神可，饮食可，大便正常，小便正常，近期体重无明显变化。", "entity_list": [{"entity_index": {"begin": 8, "end": 11}, "entity_type": "疾病和诊断", "entity": "直肠癌"}, {"entity_index": {"begin": 21, "end": 35}, "entity_type": "手术", "entity": "直肠癌根治术（DIXON术）"}, {"entity_index": {"begin": 78, "end": 95}, "entity_type": "疾病和诊断", "entity": "直肠腺癌（中低度分化），浸润溃疡型"}, {"entity_index": {"begin": 139, "end": 159}, "entity_type": "解剖部位", "entity": "肠壁一站（10个）、中间组（8个）淋巴结"}, {"entity_index": {"begin": 230, "end": 234}, "entity_type": "药物", "entity": "奥沙利铂"}, {"entity_index": {"begin": 243, "end": 247}, "entity_type": "药物", "entity": "亚叶酸钙"}, {"entity_index": {"begin": 252, "end": 255}, "entity_type": "药物", "entity": "替加氟"}, {"entity_index": {"begin": 276, "end": 277}, "entity_type": "解剖部位", "entity": "肝"}, {"entity_index": {"begin": 312, "end": 313}, "entity_type": "解剖部位", "entity": "腹"}, {"entity_index": {"begin": 314, "end": 315}, "entity_type": "解剖部位", "entity": "腹"}, {"entity_index": {"begin": 342, "end": 347}, "entity_type": "疾病和诊断", "entity": "直肠癌术后"}]}
    ...
```

### 1.6 面向试验鉴定的实体数据集

- 介绍：面向试验鉴定的命名实体数据集是由军事科学院系统工程研究院在CCKS 2020中组织的一个评测。
- 时间：2020
- 实体类型：试验要素(如：RS-24弹道导弹、SPY-1D相控阵雷达)、性能指标(如测量精度、圆概率偏差、失效距离)、系统组成(如中波红外导引头、助推器、整流罩)、任务场景(如法国海军、导弹预警、恐怖袭击)四大类。
- 数据集：400篇的标注文档
- 地址：   https://www.biendata.xyz/competition/ccks_2020_8/
- 数据格式

```s
    
```

### 1.7 BosonNLP实体数据集

- 介绍：玻森数据提供的命名实体识别数据，采用UTF-8进行编码
- 时间：2020
- 实体类型：时间、地点、人名、组织名、公司名、产品名
- 数据集：2000段落
- 地址：   https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/boson
- 数据格式

```s
    {"text": "高勇：男，中国国籍，无境外居留权，", "entity_list": [{"entity_index": {"begin": 0, "end": 2}, "entity_type": "NAME", "entity": "高勇"}, {"entity_index": {"begin": 5, "end": 9}, "entity_type": "CONT", "entity": "中国国籍"}]}
    {"text": "1966年出生，汉族，中共党员，本科学历，工程师、美国项目管理协会注册会员（PMIMember）、注册项目管理专家（PMP）、项目经理。", "entity_list": [{"entity_index": {"begin": 8, "end": 10}, "entity_type": "RACE", "entity": "汉族"}, {"entity_index": {"begin": 11, "end": 15}, "entity_type": "TITLE", "entity": "中共党员"}, {"entity_index": {"begin": 16, "end": 20}, "entity_type": "EDU", "entity": "本科学历"}, {"entity_index": {"begin": 21, "end": 24}, "entity_type": "TITLE", "entity": "工程师"}, {"entity_index": {"begin": 25, "end": 33}, "entity_type": "ORG", "entity": "美国项目管理协会"}, {"entity_index": {"begin": 33, "end": 37}, "entity_type": "TITLE", "entity": "注册会员"}, {"entity_index": {"begin": 38, "end": 47}, "entity_type": "TITLE", "entity": "PMIMember"}, {"entity_index": {"begin": 49, "end": 57}, "entity_type": "TITLE", "entity": "注册项目管理专家"}, {"entity_index": {"begin": 58, "end": 61}, "entity_type": "TITLE", "entity": "PMP"}, {"entity_index": {"begin": 63, "end": 67}, "entity_type": "TITLE", "entity": "项目经理"}]}
    ...
```

### 1.8 影视音乐书籍实体数据集

- 介绍：影视音乐书籍实体数据集
- 时间：
- 实体类型：影视、音乐、书籍
- 数据集：大约10000条，具体包括7814条训练集、977条验证集以及978条测试集。
- 地址：   https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/video_music_book_datasets
- 数据格式

```s
    {"text": "我个人前一段看过求无欲的诡案组系列，剧情不错，主要是人物特点表现的很好，人物性格很大众化", "entity_list": [{"entity_index": {"begin": 12, "end": 15}, "entity_type": "boo", "entity": "诡案组"}]}
    {"text": "本人也比较喜欢看仙侠小说，推荐几个我个人爱看的：、绝世武魂、绝世武神、追鬼龙王之极品强少、万古武帝、", "entity_list": [{"entity_index": {"begin": 25, "end": 29}, "entity_type": "boo", "entity": "绝世武魂"}, {"entity_index": {"begin": 30, "end": 34}, "entity_type": "boo", "entity": "绝世武神"}, {"entity_index": {"begin": 35, "end": 44}, "entity_type": "boo", "entity": "追鬼龙王之极品强少"}, {"entity_index": {"begin": 45, "end": 49}, "entity_type": "boo", "entity": "万古武帝"}]}
    ...
```

### 1.9 中文电子病历实体数据集

- 介绍：目前现存公开的中文电子病历标注数据十分稀缺，为了推动CNER系统在中文临床文本上的表现， CCKS在2017、2018、2019、2020都组织了面向中文电子病历的命名实体识别评测任务。

#### 1.9.1 CCKS2017数据集

- 时间：2017
- 实体类型：症状和体征、检查和检验、疾病和诊断、治疗、身体部位
- 数据集：训练集包括300个医疗记录，测试集包含100个医疗记录
- 地址：   https://www.biendata.xyz/competition/CCKS2017_2/
- 数据格式

```s

```

#### 1.9.2 CCKS2018数据集

- 时间：2018
- 实体类型：解剖部位、症状描述、独立症状、药物、手术
- 数据集：训练集包括600个医疗记录，测试集包含400个医疗记录
- 地址：  https://www.biendata.xyz/competition/CCKS2018_1
- 数据格式

```s

```

#### 1.9.3 CCKS2019数据集

- 时间：2019
- 实体类型：疾病和诊断、检查、检验、手术、药物、解剖部位
- 数据集：训练集包括1000个医疗记录，测试集包含379个医疗记录
- 地址：  https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/2020_ccks_ner
- 数据格式

```s
    {"originalText": "，患者3月前因“直肠癌”于在我院于全麻上行直肠癌根治术（DIXON术），手术过程顺利，术后给予抗感染及营养支持治疗，患者恢复好，切口愈合良好。，术后病理示：直肠腺癌（中低度分化），浸润溃疡型，面积3.5*2CM，侵达外膜。双端切线另送“近端”、“远端”及环周底部切除面未查见癌。肠壁一站（10个）、中间组（8个）淋巴结未查见癌。，免疫组化染色示：ERCC1弥漫（+）、TS少部分弱（+）、SYN（-）、CGA（-）。术后查无化疗禁忌后给予3周期化疗，，方案为：奥沙利铂150MG D1，亚叶酸钙0.3G+替加氟1.0G D2-D6，同时给与升白细胞、护肝、止吐、免疫增强治疗，患者副反应轻。院外期间患者一般情况好，无恶心，无腹痛腹胀胀不适，无现患者为行复查及化疗再次来院就诊，门诊以“直肠癌术后”收入院。   近期患者精神可，饮食可，大便正常，小便正常，近期体重无明显变化。", "entities": [{"label_type": "疾病和诊断", "overlap": 0, "start_pos": 8, "end_pos": 11}, {"label_type": "手术", "overlap": 0, "start_pos": 21, "end_pos": 35}, {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 78, "end_pos": 95}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 139, "end_pos": 159}, {"end_pos": 234, "label_type": "药物", "overlap": 0, "start_pos": 230}, {"end_pos": 247, "label_type": "药物", "overlap": 0, "start_pos": 243}, {"end_pos": 255, "label_type": "药物", "overlap": 0, "start_pos": 252}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 276, "end_pos": 277}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 312, "end_pos": 313}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 314, "end_pos": 315}, {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 342, "end_pos": 347}]}
    ...
```

#### 1.9.4 CCKS2020数据集

- 时间：2020
- 实体类型：疾病和诊断、检查、检验、手术、药物、解剖部位
- 数据集：训练集包括1050个医疗记录
- 地址：  https://www.biendata.xyz/competition/ccks_2020_2_1/
- 数据格式

```s
    对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。|||3    9    bod|||19    24    dis|||
    研究证实，细胞减少与肺内病变程度及肺内炎性病变吸收程度密切相关。|||10    10    bod|||10    13    sym|||17    17    bod|||17    22    sym|||
    ...
```

### 1.10 中文电子简历实体数据集

- 介绍：根据新浪财经网关于上市公司的高级经理人的简历摘要数据，进行筛选过滤和人工标注生成的，建于2018年。
- 时间：2018
- 实体类型：人名、国籍、籍贯、种族、专业、学位、机构、职称
- 数据集：3821条训练集、463条验证集、477条测试集
- 地址：   https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/ResumeNER
- 数据格式

```s
    {"text": "高勇：男，中国国籍，无境外居留权，", "entity_list": [{"entity_index": {"begin": 0, "end": 2}, "entity_type": "NAME", "entity": "高勇"}, {"entity_index": {"begin": 5, "end": 9}, "entity_type": "CONT", "entity": "中国国籍"}]}
    {"text": "1966年出生，汉族，中共党员，本科学历，工程师、美国项目管理协会注册会员（PMIMember）、注册项目管理专家（PMP）、项目经理。", "entity_list": [{"entity_index": {"begin": 8, "end": 10}, "entity_type": "RACE", "entity": "汉族"}, {"entity_index": {"begin": 11, "end": 15}, "entity_type": "TITLE", "entity": "中共党员"}, {"entity_index": {"begin": 16, "end": 20}, "entity_type": "EDU", "entity": "本科学历"}, {"entity_index": {"begin": 21, "end": 24}, "entity_type": "TITLE", "entity": "工程师"}, {"entity_index": {"begin": 25, "end": 33}, "entity_type": "ORG", "entity": "美国项目管理协会"}, {"entity_index": {"begin": 33, "end": 37}, "entity_type": "TITLE", "entity": "注册会员"}, {"entity_index": {"begin": 38, "end": 47}, "entity_type": "TITLE", "entity": "PMIMember"}, {"entity_index": {"begin": 49, "end": 57}, "entity_type": "TITLE", "entity": "注册项目管理专家"}, {"entity_index": {"begin": 58, "end": 61}, "entity_type": "TITLE", "entity": "PMP"}, {"entity_index": {"begin": 63, "end": 67}, "entity_type": "TITLE", "entity": "项目经理"}]}
    ...
```

### 1.11 CoNLL 2003数据集

- 介绍：1393篇英语新闻文章和909篇德语新闻文章
- 时间：2013
- 实体类型：LOC、ORG、PER、MISC
- 数据集：1393篇英语新闻文章和909篇德语新闻文章
- 地址：   https://www.clips.uantwerpen.be/conll2003/ner/
- 数据格式

```s

```

### 1.12 OntoNotes5.0 数据集

- 介绍：1745k英语、900k中文和300k阿拉伯语文本数据组成，来源于电话对话、新闻通讯社、广播新闻、广播对话和博客
- 时间：2013
- 实体类型：PERSON、ORGANIZATION和LOCATION等18个类别
- 数据集：1393篇英语新闻文章和909篇德语新闻文章
- 地址：    https://catalog.ldc.upenn.edu/ldc2013t19
- 数据格式

```s

```

### 1.13 CMeEE

- 介绍：数据集全称是Chinese Medical Entity Extraction，由“北京大学”、“郑州大学”、“鹏城实验室”和“哈尔滨工业大学（深圳）”联合提供，这是一个标准的NER识别任务
- 时间：2013
- 实体类型：疾病(dis)，临床表现(sym)，药物(dru)，医疗设备(equ)，医疗程序(pro)，身体(bod)，医学检验项目(ite)，微生物类(mic)，科室(dep)
- 数据集：
- 地址：    https://tianchi.aliyun.com/specials/promotion/2021chinesemedicalnlpleaderboardchallenge
- 数据格式

```s
[
  {
    "text": "（5）房室结消融和起搏器植入作为反复发作或难治性心房内折返性心动过速的替代疗法。",
    "entities": [
      {
        "start_idx": 3,
        "end_idx": 7,
        "type": "pro",
        "entity": "房室结消融"
      },
      {
        "start_idx": 9,
        "end_idx": 13,
        "type": "pro",
        "entity": "起搏器植入"
      },
      {
        "start_idx": 16,
        "end_idx": 33,
        "type": "dis",
        "entity": "反复发作或难治性心房内折返性心动过速"
      }
    ]
  },...
]
```

## 二、实体关系抽取数据集

### 2.11 CMeIE

- 介绍：数据集全称是Chinese Medical Information Extraction，与CMeEE的数据提供方一样。这是一个关系抽取任务，共包括53类关系类型（具体类型参加官网介绍），从关系种类数量来看，这是一个比较难的任务。这个任务与传统的关系抽取任务有两处不同： 1. 预测阶段并没有事先给出要判定的实体，输入就是原始的文本，因此选手需要同时处理实体识别和关系抽取，可以看作是一个端对端的关系抽取任务；2. 训练数据中的实体并没有给出具体的下标，如果一个实体在句子中多次出现，难点是无法得知关系中的实体具体是指哪一次出现的实体。
- 时间：2020
- 实体关系类型：53_schema
- 数据集：14,339 training set data, 3,585 validation set data, 4,482 test set data
- 地址：      https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414&lang=en-us
- 数据格式

```s
{  
  "text": "慢性胰腺炎@ ###低剂量放射 自1964年起，有几项病例系列报道称外照射 (5-50Gy) 可以有效改善慢性胰腺炎患者的疼痛症状。慢性胰腺炎@从概念上讲，外照射可以起到抗炎和止痛作用，并且已经开始被用于非肿瘤性疼痛的治疗。", 
  "spo_list": [ 
    { 
      "predicate": "放射治疗", 
      "subject": "慢性胰腺炎", 
      "subject_type": "疾病", 
      "object": { "@value": "外照射" }, 
      "object_type": { "@value": "其他治疗" } 
    }, 
    { 
      "predicate": "放射治疗", 
      "subject": "非肿瘤性疼痛", 
      "subject_type": "疾病", 
      "object": { "@value": "外照射" }, 
      "object_type": { "@value": "其他治疗" } 
      }
    }
  ] 
}
```

> 53_schemas.json
```s
{"subject_type": "疾病", "predicate": "预防", "object_type": "其他"}
{"subject_type": "疾病", "predicate": "阶段", "object_type": "其他"}
{"subject_type": "疾病", "predicate": "就诊科室", "object_type": "其他"}
{"subject_type": "其他", "predicate": "同义词", "object_type": "其他"}
{"subject_type": "疾病", "predicate": "辅助治疗", "object_type": "其他治疗"}
...
```

### 2.10 DocRED文档级实体关系数据集

- 介绍：基于维基百科的文档级关系抽取数据集
- 时间：2019
- 实体关系类型：命名实体提及、核心参考信息、句内和句间关系以及支持证据。关系类型涉及科学、艺术、时间、个人生活在内的96种Wikidata关系类型。
- 数据集：在5053个维基百科文档上进行标注，包含132375个实体和56354个关系事实。
- 地址：      https://github.com/thunlp/DocRED
- 数据格式

```s
{
  'title',
  'sents':     [
                  [word in sent 0],
                  [word in sent 1]
               ]
  'vertexSet': [
                  [
                    { 'name': mention_name, 
                      'sent_id': mention in which sentence, 
                      'pos': postion of mention in a sentence, 
                      'type': NER_type}
                    {anthor mention}
                  ], 
                  [anthoer entity]
                ]
  'labels':   [
                {
                  'h': idx of head entity in vertexSet,
                  't': idx of tail entity in vertexSet,
                  'r': relation,
                  'evidence': evidence sentences' id
                }
              ]
}
```

### 2.9 Chinese Literature Text文档级实体关系数据集

- 介绍：面向中文文学的一个实体关系数据集
- 时间：2019
- 实体关系类型：物体、人名、地名、时间名、容量名、组织和摘要共7类实体，位于、部分、家庭、概括、社会、拥有、使用、制造、邻接等9类实体关系
- 数据集：共计726篇文章，29096句话，超过100000个字符。训练集695篇，验证集58篇、测试集84篇。
- 地址：      https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset
- 数据格式

> ner train.txt
```s
记 O
得 O
小 B_Time
时 I_Time
候 I_Time
， O
妈 B_Person
妈 I_Person
说 O
起 O
哪 O
个 O
典 O
型 O
败 B_Person
家 I_Person
子 I_Person
形 O
象 O
， O
挂 O
在 O
嘴 O
边 O
的 O
就 O
是 O
那 B_Person
人 I_Person
吃 O
喝 O
嫖 O
赌 O
瘾 O
五 O
毒 O
俱 O
全 O
。 O
...
```

> relation_extraction

> txt
```s
清明是人们祭扫先人，怀念追思的日子。正如宋代诗人高翥所云“南北山头多墓田，清明祭扫各纷然。纸灰飞作白蝴蝶，泪血染成红杜鹃
”。凡清明之时，总是屡屡哀思涌上心头，对母亲怀念的情愫越发细腻绵长。
...
```

> ann
```s
T1	Person-Name 20 26	宋代诗人高翥
T2	Person-Pronoun 108 109	我
...
```

### 2.8 人物实体关系数据集

- 介绍：CCKS2019中的一个层级关系分类任务
- 时间：2019
- 实体关系类型：三大类(亲属关系、社交关系、师生关系)，四中类(配偶、血亲、姻亲、友谊）、35小类(现夫、前妻)种关系类型
- 数据集：3841条验证集、287351条训练集以及77092条测试集句子
- 地址：     https://github.com/SUDA-HLT/IPRE
- 数据格式

> bag_relation_train.txt
```s
TRAIN_BAG_ID_000001	金泰熙	金东	TRAIN_SENT_ID_000001	0
TRAIN_BAG_ID_000002	辛文山	林散之	TRAIN_SENT_ID_000002	0
...
```

> bag_relation_train.txt
```s
TRAIN_SENT_ID_000001	0
TRAIN_SENT_ID_000002	0
...
```


> sent_train_1.txt
```s
TRAIN_SENT_ID_000001	金泰熙	金东	韩国 梦想 演唱会 第十届 2004 年 : MC : 金泰熙 ， 金东 万
TRAIN_SENT_ID_000002	辛文山	林散之	林散之 先生 等 当代 名家 对 辛文山 先生 的 书法 均 有 精辟 的 点评 ， 对 书法 爱好者 自学 书法 有 较 高 的 参考价值 。
...
```

> schema.json
```s
NA	0
人物关系/亲属关系/配偶/丈夫/现夫	1
人物关系/亲属关系/配偶/丈夫/前夫	2
人物关系/亲属关系/配偶/丈夫/未婚夫	3
...
```

### 2.7 COAE2016实体关系数据集

- 介绍：CAOE2016 task3任务中用到的一个关系数据集
- 时间：2016
- 实体关系类型：关系类别包括出生日期、出生地、毕业院校、配偶、子女、高管、员工数、创始人、总部、其他共十类关系。
- 数据集：包含988个训练数据和483个测试数据
- 地址：   NRE\chinese
- 数据格式

> 训练数据
> schema.json
```s
{'NA': 0, '/人物/其它/职业': 1, '/人物/组织/毕业于': 2, '/人物/其它/民族': 3, '/地点/地点/毗邻': 4, '/人物/地点/出生地': 5, '/人物/地点/国籍': 6, '/人物/组织/属于': 7, '/人物/人物/家庭成员': 8, '/组织/组织/周边': 9, '/组织/地点/位于': 10, '/地点/地点/包含': 11, '/地点/组织/景点': 12, '/地点/人物/相关人物': 13, '/地点/地点/首都': 14, '/组织/人物/校长': 15, '/组织/人物/创始人': 16, '/地点/其它/气候': 17, '/组织/人物/领导人': 18, '/组织/人物/拥有者': 19, '/地点/地点/位于': 20, '/人物/人物/社交关系': 21, '/人物/地点/居住地': 22}
```

### 2.6 DuIE2.0实体关系数据集

- 介绍：业界规模最大的基于schema的中文关系抽取数据集，来自百度百科、百度贴吧和百度信息流文本。
- 时间：2020
- 实体关系类型：包含超过43万三元组数据、21万中文句子及48个预定义的关系类型。
- 数据集：包括171135个训练集、21055个测试数据，外加80184条混淆数据。
- 地址：    https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/DuIE
- 数据格式

> 训练数据
```s
    {"text": "吴宗宪遭服务生种族歧视, 他气呛: 我买下美国都行!艺人狄莺与孙鹏18岁的独子孙安佐赴美国读高中，没想到短短不到半年竟闹出校园安全事件被捕，因为美国正处于校园枪击案频传的敏感时机，加上国外种族歧视严重，外界对于孙安佐的情况感到不乐观 吴宗宪今（30）日录影前谈到美国民情，直言国外种族歧视严重，他甚至还被一名墨西哥裔的服务生看不起，让吴宗宪气到喊：「我是吃不起是不是", "spo_list": [{"predicate": "父亲", "object_type": {"@value": "人物"}, "subject_type": "人物", "object": {"@value": "孙鹏"}, "subject": "孙安佐"}, {"predicate": "母亲", "object_type": {"@value": "人物"}, "subject_type": "人物", "object": {"@value": "狄莺"}, "subject": "孙安佐"}, {"predicate": "丈夫", "object_type": {"@value": "人物"}, "subject_type": "人物", "object": {"@value": "孙鹏"}, "subject": "狄莺"}, {"predicate": "妻子", "object_type": {"@value": "人物"}, "subject_type": "人物", "object": {"@value": "狄莺"}, "subject": "孙鹏"}]}
    ...
```

> schema.json
```s
    {"object_type": {"@value": "学校"}, "predicate": "毕业院校", "subject_type": "人物"}
    {"object_type": {"@value": "人物"}, "predicate": "嘉宾", "subject_type": "电视综艺"}
    {"object_type": {"inWork": "影视作品", "@value": "人物"}, "predicate": "配音", "subject_type": "娱乐人物"}
    {"object_type": {"@value": "歌曲"}, "predicate": "主题曲", "subject_type": "影视作品"}
    {"object_type": {"@value": "人物"}, "predicate": "代言人", "subject_type": "企业/品牌"}
    {"object_type": {"@value": "音乐专辑"}, "predicate": "所属专辑", "subject_type": "歌曲"}
    {"object_type": {"@value": "人物"}, "predicate": "父亲", "subject_type": "人物"}
    {"object_type": {"@value": "人物"}, "predicate": "作者", "subject_type": "图书作品"}
    {"object_type": {"inArea": "地点", "@value": "Date"}, "predicate": "上映时间", "subject_type": "影视作品"}
    {"object_type": {"@value": "人物"}, "predicate": "母亲", "subject_type": "人物"}
    {"object_type": {"@value": "Text"}, "predicate": "专业代码", "subject_type": "学科专业"}
    {"object_type": {"@value": "Number"}, "predicate": "占地面积", "subject_type": "机构"}
    {"object_type": {"@value": "Text"}, "predicate": "邮政编码", "subject_type": "行政区"}
    {"object_type": {"inArea": "地点", "@value": "Number"}, "predicate": "票房", "subject_type": "影视作品"}
    {"object_type": {"@value": "Number"}, "predicate": "注册资本", "subject_type": "企业"}
    {"object_type": {"@value": "人物"}, "predicate": "主角", "subject_type": "文学作品"}
    {"object_type": {"@value": "人物"}, "predicate": "妻子", "subject_type": "人物"}
    {"object_type": {"@value": "人物"}, "predicate": "编剧", "subject_type": "影视作品"}
    {"object_type": {"@value": "气候"}, "predicate": "气候", "subject_type": "行政区"}
    {"object_type": {"@value": "人物"}, "predicate": "歌手", "subject_type": "歌曲"}
    {"object_type": {"inWork": "作品", "onDate": "Date", "@value": "奖项", "period": "Number"}, "predicate": "获奖", "subject_type": "娱乐人物"}
    {"object_type": {"@value": "人物"}, "predicate": "校长", "subject_type": "学校"}
    {"object_type": {"@value": "人物"}, "predicate": "创始人", "subject_type": "企业"}
    {"object_type": {"@value": "城市"}, "predicate": "首都", "subject_type": "国家"}
    {"object_type": {"@value": "人物"}, "predicate": "丈夫", "subject_type": "人物"}
    {"object_type": {"@value": "Text"}, "predicate": "朝代", "subject_type": "历史人物"}
    {"object_type": {"inWork": "影视作品", "@value": "人物"}, "predicate": "饰演", "subject_type": "娱乐人物"}
    {"object_type": {"@value": "Number"}, "predicate": "面积", "subject_type": "行政区"}
    {"object_type": {"@value": "地点"}, "predicate": "总部地点", "subject_type": "企业"}
    {"object_type": {"@value": "地点"}, "predicate": "祖籍", "subject_type": "人物"}
    {"object_type": {"@value": "Number"}, "predicate": "人口数量", "subject_type": "行政区"}
    {"object_type": {"@value": "人物"}, "predicate": "制片人", "subject_type": "影视作品"}
    {"object_type": {"@value": "Number"}, "predicate": "修业年限", "subject_type": "学科专业"}
    {"object_type": {"@value": "城市"}, "predicate": "所在城市", "subject_type": "景点"}
    {"object_type": {"@value": "人物"}, "predicate": "董事长", "subject_type": "企业"}
    {"object_type": {"@value": "人物"}, "predicate": "作词", "subject_type": "歌曲"}
    {"object_type": {"@value": "作品"}, "predicate": "改编自", "subject_type": "影视作品"}
    {"object_type": {"@value": "企业"}, "predicate": "出品公司", "subject_type": "影视作品"}
    {"object_type": {"@value": "人物"}, "predicate": "导演", "subject_type": "影视作品"}
    {"object_type": {"@value": "人物"}, "predicate": "作曲", "subject_type": "歌曲"}
    {"object_type": {"@value": "人物"}, "predicate": "主演", "subject_type": "影视作品"}
    {"object_type": {"@value": "人物"}, "predicate": "主持人", "subject_type": "电视综艺"}
    {"object_type": {"@value": "Date"}, "predicate": "成立日期", "subject_type": "机构"}
    {"object_type": {"@value": "Text"}, "predicate": "简称", "subject_type": "机构"}
    {"object_type": {"@value": "Number"}, "predicate": "海拔", "subject_type": "地点"}
    {"object_type": {"@value": "Text"}, "predicate": "号", "subject_type": "历史人物"}
    {"object_type": {"@value": "国家"}, "predicate": "国籍", "subject_type": "人物"}
    {"object_type": {"@value": "语言"}, "predicate": "官方语言", "subject_type": "国家"}
```

### 2.5 NYT10实体关系数据集

- 介绍：在基于远程监督的关系抽取任务上最常用的数据集，由NYT corpus 在2010年基于Freebase远程监督得到的
- 时间：2010
- 实体关系类型：founders、place_of_birth在内的53种关系（包括一种NA），存在一定的噪声。
- 数据集：466876条训练集、55167条验证集以及172448条测试集。
- 地址：    https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_nyt10.sh
- 数据格式

```s

```

### 2.4 Wiki80实体关系数据集

- 介绍：从数据集FewRel上提取的一个关系数据集
- 时间：
- 实体关系类型：包含location、part of、follows等80种关系，每种关系个数均为700，共56000个样本。
- 数据集：50400条训练集、5600条验证集
- 地址：   https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_wiki80.sh
- 数据格式

```s

```

### 2.3 FewRel实体关系数据集

- 介绍：清华大学于2018年发布的精标注关系抽取数据集，是当前规模最大的中文实体关系数据集
- 时间：2018
- 实体关系类型：100个关系类别、70,000个关系实例
- 数据集：每句的平均长度为24.99，一共出现 124,577 个不同的单词/符号。
- 地址：  https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_fewrel.sh
- 数据格式

```s

```

### 2.2 SemEval实体关系数据集

- 介绍：2010年国际语义评测大会中Task8任务所使用的数据集
- 时间：2010
- 实体关系类型：Cause-Effect(因果关系)、Instrument-Agency(操作、使用关系)、Product-Producer(产品-生产者关系)、 Content-Container(空间包含关系)、Entity-Origin(起源关系)、Entity-Destination(导向关系)、 Component-Whole(组件-整体关系)、Member-Collection(成员-集合关系)、Message-Topic(主题关系)等10类关系。
- 数据集：8000个训练样本，2717个测试样本
- 地址：  https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_semeval.sh
- 数据格式

```s

```

### 2.1 ACE实体关系数据集

- 介绍：包括英语，阿拉伯语和中文三部分数据，分成广播新闻和新闻专线两部分
- 时间：2005
- 实体关系类型：ART、Gen-affiliation在内的6种关系类型，Employment、Founder、Ownership在内的额18种子关系类型。
- 数据集：451个文档和5 702个关系实例。ACE2005中文数据集包括633个文档、307991个字符
- 地址：     https://catalog.ldc.upenn.edu/byproject
- 数据格式

```s

```

## 参考

1. [数据资源：常用12类实体识别、10类关系抽取数据集的梳理与思考](https://mp.weixin.qq.com/s/fjRcDANDGMFh9eH_YyO03w)