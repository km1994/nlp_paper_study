# 【关于 PURE 源码分析】 那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 
> 论文：A Frustratingly Easy Approach for Joint Entity and Relation Extraction</br>
> 论文地址：chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F2010.12812.pdf</br>
> github : </br>
>   官方：https://github.com/princeton-nlp/PURE </br>
>   复现版：https://github.com/suolyer/PyTorch_BERT_Pipeline_IE</br>

## 一、requirement 

- python==3.6+
- scikit-learn              0.24.2
- tokenizers                0.10.3 
- torch                     1.9.0+cu111  
- torchaudio                0.9.0  
- torchvision               0.10.0+cu111  
- tqdm                      4.62.2       
- transformers              4.9.2    

## 二、数据格式介绍

### 2.1 CMeIE 数据介绍

- CMeIE_train.json

```json
    {"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    {"text": "类风湿关节炎@尺侧偏斜是由于MCP关节炎症造成的。", "spo_list": [{"Combined": false, "predicate": "临床表现", "subject": "MCP关节炎症", "subject_type": "疾病", "object": {"@value": "尺侧偏斜"}, "object_type": {"@value": "症状"}}]}
    {"text": "唇腭裂@ ### 腭瘘 | 存在差异 | 低 大约 10% 至 20% 颚成形术发生腭瘘。 唇腭裂@腭瘘发生机率与婴儿伤口，营养状况，外科技术和其他因素相关。", "spo_list": [{"Combined": true, "predicate": "风险评估因素", "subject": "腭瘘", "subject_type": "疾病", "object": {"@value": "婴儿伤口"}, "object_type": {"@value": "社会学"}}, {"Combined": true, "predicate": "风险评估因素", "subject": "腭瘘", "subject_type": "疾病", "object": {"@value": "营养状况"}, "object_type": {"@value": "社会学"}}, {"Combined": true, "predicate": "风险评估因素", "subject": "腭瘘", "subject_type": "疾病", "object": {"@value": "外科技术"}, "object_type": {"@value": "社会学"}}]}
    ...
```

- 53_schemas.json

```json
    {"subject_type": "疾病", "predicate": "预防", "object_type": "其他"}
    {"subject_type": "疾病", "predicate": "阶段", "object_type": "其他"}
    {"subject_type": "疾病", "predicate": "就诊科室", "object_type": "其他"}
    {"subject_type": "其他", "predicate": "同义词", "object_type": "其他"}
    {"subject_type": "疾病", "predicate": "辅助治疗", "object_type": "其他治疗"}
    ...
```

### 2.2 lic2020 数据介绍

- train_data.json

```json
    {"text": "《邪少兵王》是冰火未央写的网络小说连载于旗峰天下", "spo_list": [{"predicate": "作者", "object_type": {"@value": "人物"}, "subject_type": "图书作品", "object": {"@value": "冰火未央"}, "subject": "邪少兵王"}]}
    {"text": "GV-971由中国海洋大学、中国科学院上海药物研究所（下称“上海药物所”）和上海绿谷制药有限公司（下称“绿谷制药”）联合研发，不同于传统靶向抗体药物，GV-971是从海藻中提取的海洋寡糖类分子", "spo_list": [{"predicate": "简称", "object_type": {"@value": "Text"}, "subject_type": "机构", "object": {"@value": "上海药物所"}, "subject": "中国科学院上海药物研究所"}]}
    ...
```

- schema.json

```json
    {"object_type": {"@value": "学校"}, "predicate": "毕业院校", "subject_type": "人物"}
    {"object_type": {"@value": "人物"}, "predicate": "嘉宾", "subject_type": "电视综艺"}
    {"object_type": {"inWork": "影视作品", "@value": "人物"}, "predicate": "配音", "subject_type": "娱乐人物"}
    ...
```

## 三、运行

- 模型训练

```python
    python train.py
```


## 四、代码分析

### 4.1 数据预处理模块

#### 4.1.1 train.py 函数调用

1. train.py 函数调用

```python
from data_preprocessing import data_prepro
...
def train():
    train_data = data_prepro.yield_data(args.train_path)
    test_data = data_prepro.yield_data(args.dev_path)
    ...

if __name__=='__main__':
    train()
```

2. data_preprocessing/data_prepro.py yield_data 函数介绍

```python
...
def yield_data(file_path):
    tmp = MyDataset(load_data(file_path))
    return DataLoader(tmp, batch_size=args.batch_size, shuffle=True)
...
```

#### 4.1.2 关联函数调用

1. data_preprocessing/data_prepro.py load_data 函数介绍

```python
# 功能：数据加载
def load_data(file_path):
    '''
        功能：数据加载
        input:
            file_path:String          输入文件
        return:
            result:List()             结果数据
    '''
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentences = []
        spo_lists = []
        result=[]
        for line in tqdm(lines):
            data = json.loads(line)
            text = data['text']
            one_spo_list=[]
            for spo in data['spo_list']:
                s=spo['subject']
                p=spo['predicate']
                tmp_ob_type=[v for k,v in spo['object_type'].items()]
                tmp_ob=[v for k,v in spo['object'].items()]
                for i in range(len(tmp_ob)):
                    p_o=p+'|'+tmp_ob_type[i]
                    one_spo_list.append((s,p_o,tmp_ob[i]))
            sentences.append(text)
            spo_lists.append(one_spo_list)
            result.append({'text':text,'spo_list':one_spo_list})
        return result
```

> 输入数据

```s
    {"text": "《邪少兵王》是冰火未央写的网络小说连载于旗峰天下", "spo_list": [{"predicate": "作者", "object_type": {"@value": "人物"}, "subject_type": "图书作品", "object": {"@value": "冰火未央"}, "subject": "邪少兵王"}]}
    {"text": "GV-971由中国海洋大学、中国科学院上海药物研究所（下称“上海药物所”）和上海绿谷制药有限公司（下称“绿谷制药”）联合研发，不同于传统靶向抗体药物，GV-971是从海藻中提取的海洋寡糖类分子", "spo_list": [{"predicate": "简称", "object_type": {"@value": "Text"}, "subject_type": "机构", "object": {"@value": "上海药物所"}, "subject": "中国科学院上海药物研究所"}]}
    ...
```

> 输出数据

```s
[
    {'text': '产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。', 'spo_list': [('产后抑郁症', '鉴别诊断|疾病', '轻度情绪失调')]}, {'text': '类风湿关节炎@尺侧偏斜是由于MCP关节炎症造成的。', 'spo_list': [('MCP关节炎症', '临床表现|症状', '尺侧偏斜')]}, 
    {'text': '唇腭裂@ ### 腭瘘 | 存在差异 | 低 大约 10% 至 20% 颚成形术发生腭瘘。 唇腭裂@腭瘘发生机率与婴儿伤口，营养状况，外科技术和其他因素相关。', 'spo_list': [('腭瘘', '风险 评估因素|社会学', '婴儿伤口'), ('腭瘘', '风险评估因素|社会学', '营养状况'), ('腭瘘', '风险评估因素|社会学', '外科技术')]}
    ,...
]
...
```

2. 