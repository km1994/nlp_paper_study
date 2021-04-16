#coding=utf-8
import numpy as np
import json
import string
import os
import re
from collections import defaultdict

# 将人民日报数据集进行转换
def transfer_data_0(source_file, target_file):
    '''
    人民日报数据格式：
    1	迈向	vt	O	_
    2	充满	vt	O	_
    3	希望	n	O	_
    4	的	ud	O	_
    5	新	a	O	_
    6	世纪	n	O	_
    7	——	wp	O	_
    8	一九九八年新年	t	DATE	_
    9	讲话	n	O	_
    10	(	wkz	O	_
    '''
    with open(source_file) as f, open(
            target_file, "w+", encoding="utf-8") as g:
        text = ""
        entity_list = []  # {"entity_index": {"begin": 21, "end": 25}, "entity_type": "影视作品", "entity": "喜剧之王"}
        lines = 0
        for word_line in f:
            if word_line != "\n":  # 是句子的词
                # print(word_line)
                word_split = word_line.strip().split("\t")
                # print(word_split)
                if word_split[3] != "O":
                    entity_list.append({"entity_index": {"begin": len(text), "end": len(text + word_split[1])},
                                        "entity_type": word_split[3], "entity": word_split[1]})
                text += word_split[1]
            else:  # 句子的结尾
                g.write(json.dumps({"text": text, "entity_list": entity_list}, ensure_ascii=False) + "\n")
                lines += 1
                text = ""
                entity_list = []
                if lines == 1000:
                    break
    print("共有{}行".format(lines))


# 将平常ner标注(微博、微软)数据转化为项目所需数据格式
def transfer_data_1(source_file, target_file):
    # 同时满足BIEO标注、BIO标注和BMESO标注
    '''
    将
    男	B-PER.NOM /// B-PER
    女	B-PER.NOM
    必	O
    看	O
    的	O
    微	O
    博	O
    花	O
    心	O

    我	O
    参	O
    与	O
    了	O
    南	B-GPE.NAM
    都	I-GPE.NAM
    标注类型转化为要求的数据格式
    '''
    with open(source_file, errors="ignore") as f, open(target_file, "w+", encoding="utf-8") as g:
        text = ""
        entity_list = [] # {"entity_index": {"begin": 21, "end": 25}, "entity_type": "影视作品", "entity": "喜剧之王"}
        lines = 0
        words_start = 0 # 词的开始边界
        words_end = 0 # 词的结束边界
        words_bool = None # 是否存在未加入的新词，存在的话设置为词的类型，默认没有
        for word_line in f:
            word_line = word_line.strip()
            word_split = word_line.strip().split(" ")
            if '' in word_split:
                word_split.remove('')
            if word_split: # 是句子的词
                if len(word_split) == 1:
                    word_split.insert(0, "、")
                # print(word_split)
                if (word_split[1].startswith("B") or word_split[1].startswith("S")) and not word_split[1].endswith("NOM"):
                    if words_bool:
                        entity_list.append({"entity_index": {"begin": words_start, "end": words_start + words_end},
                                            "entity_type": words_bool,
                                            "entity": text[words_start:words_start + words_end]})
                    words_start = len(text)
                    words_end = 1
                    if "." in word_split[1]:
                        words_bool = word_split[1][2:word_split[1].rfind(".")+1]
                    else:
                        words_bool = word_split[1][2:]
                elif (word_split[1].startswith("M") or word_split[1].startswith("I") or word_split[1].startswith("E")) and not word_split[1].endswith("NOM"):
                    words_end += 1
                elif word_split[1] == "O" and words_bool:
                    entity_list.append({"entity_index": {"begin": words_start, "end": words_start + words_end},
                                        "entity_type": words_bool,
                                        "entity": text[words_start:words_start + words_end]})
                    words_bool = None
                text += word_split[0]
            else: # 句子的结尾
                if words_bool:
                    entity_list.append({"entity_index": {"begin": words_start, "end": words_start + words_end},
                                        "entity_type": words_bool,
                                        "entity": text[words_start:words_start + words_end]})
                    words_bool = None
                g.write(json.dumps({"text":text,"entity_list":entity_list}, ensure_ascii=False) + "\n")
                lines += 1
                text = ""
                entity_list = []
                # if lines == 1000:
                #     break
    print("共有{}行".format(lines))
#
# transfer_data_1("/home/liguocai/model_py36/data_diversity/product_testdata_kg/open_ner_data/source_data/ChineseNLPCorpus/NER/MSRA/dh_msra.txt",
#                 "/home/liguocai/model_py36/data_diversity/product_testdata_kg/open_ner_data/msra_1000.txt")
# transfer_data_1("/home/liguocai/model_py36/data_diversity/product_testdata_kg/open_ner_data/video_music_book_datasets/data/train.txt",
#                 "/home/liguocai/model_py36/data_diversity/product_testdata_kg/open_ner_data/video_music_book_datasets/train.txt")
# transfer_data_1("/home/liguocai/model_py36/data_diversity/product_testdata_kg/open_ner_data/video_music_book_datasets/data/valid.txt",
#                 "/home/liguocai/model_py36/data_diversity/product_testdata_kg/open_ner_data/video_music_book_datasets/dev.txt")
# transfer_data_1("/home/liguocai/model_py36/data_diversity/product_testdata_kg/open_ner_data/video_music_book_datasets/data/test.txt",
#                 "/home/liguocai/model_py36/data_diversity/product_testdata_kg/open_ner_data/video_music_book_datasets/test.txt")
# transfer_data_1("./ResumeNER/train.char.bmes", "./ResumeNER/train.txt")
# transfer_data_1("./ResumeNER/dev.char.bmes", "./ResumeNER/dev.txt")
# transfer_data_1("./ResumeNER/test.char.bmes", "./ResumeNER/test.txt")



# boson ner数据格式转化
def transfer_data_2(source_file, target_file):
    '''
    boson数据格式：
    完成!!!!!!!!!!给大家看看 {{time:今天}}{{person_name:吕小珊}}要交大家 新手也可以简单上手!!! 上学也不会觉得奇怪的妆感喔^^ 大家加油喔~~!!!!!你的喜欢
    会是{{person_name:吕小珊}} 最你的喜欢 会是{{person_name:吕小珊}} 最大的动力唷~~!!! 谢谢大家~~ 大的动力唷~~!!! 谢谢大家~~
    '''
    p = re.compile("({{.*?:.*?}})")
    p_ = re.compile("{{.*?:(.*?)}}")
    length = 0
    with open(source_file) as f, open(target_file, "w+", encoding="utf-8") as g:
        for s in f:
            total_de = 0
            entity_list = []

            for item1, item2 in zip(p.finditer(s), p_.findall(s)):
                # 替换
                start = item1.start() - total_de
                ss = s[start:item1.end() - total_de]
                total_de += len(ss) - len(item2)
                s = s.replace(ss, item2, 1)
                item1.start() - total_de
                entity_list.append({"entity_index": {"begin": start, "end": start + len(item2)},
                                    "entity_type": ss[2:len(ss) - 3 - len(item2)], "entity": item2})

            g.write(json.dumps({"text": s, "entity_list": entity_list}, ensure_ascii=False)+"\n")
            length += 1
            if length == 1000:
                break
        print("共有{}行".format(length))
# transfer_data_1("/home/liguocai/model_py36/data_diversity/product_testdata_kg/open_ner_data/source_data/ChineseNLPCorpus/NER/boson/origindata.txt",
#                 "/home/liguocai/model_py36/data_diversity/product_testdata_kg/open_ner_data/boson_1000.txt")


# clue数据集转化
def transfer_data_3(source_file, target_file):
    '''
    源数据：
    {"text": "她写道：抗战胜利时我从重庆坐民联轮到南京，去中山陵瞻仰，也到秦淮河去过。然后就去北京了。", "label": {"address": {"重庆": [[11, 12]], "南京": [[18, 19]],
    "北京": [[40, 41]]}, "scene": {"中山陵": [[22, 24]], "秦淮河": [[30, 32]]}}}
    '''
    with open(source_file) as f, open(target_file, "w+", encoding="utf-8") as g:
        length = 0
        for line in f:
            line_json = json.loads(line)
            text = line_json['text']
            entity_list = []

            if "label" in line_json.keys():
                for label, e in line_json['label'].items():
                    for e_name, e_index in e.items():
                        entity_list.append({"entity_index": {"begin": e_index[0][0], "end":  e_index[0][1]+1},
                                            "entity_type": label, "entity": e_name})

            g.write(json.dumps({"text": text, "entity_list": entity_list}, ensure_ascii=False) + "\n")
            length += 1
            if length == 1000:
                break

        print("共有{}行".format(length))

# transfer_data_3('./open_ner_data/cluener_public/dev.json', './open_ner_data/cluener_public/dev.txt')
# transfer_data_3('./open_ner_data/cluener_public/train.json', './open_ner_data/cluener_public/train_1000.txt')
# transfer_data_3('./open_ner_data/cluener_public/test.json', './open_ner_data/cluener_public/test.txt')

# 将brat标注的文件转化为所需格式
def transfer_data_4(source_file, test=False):
    """
    T1      DRUG_EFFICACY 1 5       补肾益肺
    T2      DRUG_EFFICACY 6 10      益精助阳
    T3      DRUG_EFFICACY 11 15     益气定喘
    T4      SYMPTOM 23 27   精神倦怠
    T5      SYNDROME 35 37  阴虚
    T6      SYMPTOM 37 39   咳嗽
    T7      SYMPTOM 39 41   体弱
    """

    lines = 0

    map_dict = {"DRUG":"药品",
                "DRUG_INGREDIENT":"药物成分",
                "DISEASE":"疾病",
                "SYMPTOM":"症状",
                "SYNDROME":"证候",
                "DISEASE_GROUP":"疾病分组",
                "FOOD":"食物",
                "FOOD_GROUP":"食物分组",
                "PERSON_GROUP":"人群",
                "DRUG_GROUP":"药品分组",
                "DRUG_DOSAGE":"药物剂型",
                "DRUG_TASTE":"药物性味",
                "DRUG_EFFICACY":"中药功效"}

    if not test:
        file_list = []
        for file_name in os.listdir(source_file):
            if file_name.endswith(".ann"):
                file_list.append(file_name[:-3])
        with open(source_file[:source_file.rfind("/")+1] + "train.txt", "w+", encoding="utf-8") as f:
            for file_name in file_list:
                with open(os.path.join(source_file,file_name+"ann")) as w, open(os.path.join(source_file,file_name+"txt")) as g:
                    text = g.read()
                    entity_list = []
                    for line in w:
                        _, entity_type, begin, end, entity = line.strip().split()
                        entity_type, begin, end = map_dict[entity_type], int(begin), int(end)
                        entity_list.append({"entity_index": {"begin": begin, "end":  end},
                                            "entity_type": entity_type, "entity": entity})
                    f.write(json.dumps({"text": text, "entity_list": entity_list}, ensure_ascii=False) + "\n")
                    lines += 1
    else:
        with open(source_file[:source_file.rfind("/") + 1] + "test.txt", "w+", encoding="utf-8") as f:
            for file in os.listdir(source_file):
                with open(os.path.join(source_file,file)) as g:
                    text = g.read()
                    f.write(json.dumps({"text": text, "entity_list": []}, ensure_ascii=False) + "\n")
                    lines += 1

    print("共有数据{}行".format(lines))

# transfer_data_4("./open_ner_data/tianchi_yiyao/train", test=False)
# transfer_data_4("./open_ner_data/tianchi_yiyao/chusai_xuanshou", test=True)

# 依渡云数据集格式转化
def transfer_data_5(source_file, target_file):
    """
    {"originalText": "，患者7月前因“下腹腹胀伴反酸”至我院就诊，完善相关检查，诊断“胃体胃窦癌(CT4N2M0,IIIB期)”，
    建议先行化疗，患者及家属表示理解同意 ，遂于2015-5-26、2015-06-19、2015-07-13分别予XELOX
    (希罗达 1250MG BID PO D1-14+奥沙利铂150MG IVDRIP Q3W)化疗三程,过程顺利，无明显副反应，
    后于2015-08-24在全麻上行胃癌根治术（远端胃大切），术程顺利，术后预防感染支持对症等处理。，术后病理示：
    胃中至低分化管状腺癌（LAUREN，分型：肠型），浸润至胃壁浆膜上层，可见神经束侵犯，未见明确脉管内癌栓；
    肿瘤消退分级（MANDARD），：TRG4；网膜组织未见癌；LN（-）；YPT3N0M0，IIA期。术后恢复可，于2015-10-10、
    开始采用XELOX化疗方案化疗（奥沙利铂150MG Q3W IVDRIP+卡培他滨1250MGBID*14天）一程，过程顺利。
    现为行上程化疗来我院就诊，拟“胃癌综合治疗后” 收入我科。自下次出院以来，患者精神可，食欲尚可，大小便正常，
    体重无明显上降。", "entities":
    [{"end_pos": 10, "label_type": "解剖部位", "overlap": 0, "start_pos": 8},
    {"end_pos": 11, "label_type": "解剖部位", "overlap": 0, "start_pos": 10},
    {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 32, "end_pos": 52},
    {"end_pos": 118, "label_type": "药物", "overlap": 0, "start_pos": 115},
    {"end_pos": 143, "label_type": "药物", "overlap": 0, "start_pos": 139},
    {"label_type": "手术", "overlap": 0, "start_pos": 193, "end_pos": 206},
    {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 233, "end_pos": 257},
    {"label_type": "解剖部位", "overlap": 0, "start_pos": 261, "end_pos": 262},
    {"end_pos": 374, "label_type": "药 物", "overlap": 0, "start_pos": 370},
    {"end_pos": 395, "label_type": "药物", "overlap": 0, "start_pos": 391},
    {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 432, "end_pos": 439}]}
    """
    with open(source_file, encoding="utf-8-sig") as f, open(target_file, "w+", encoding="utf-8") as g:
        length = 0
        error = 0
        for line in f:
            try:
                line_json = json.loads(line)
                entity_list = []
                text = line_json["originalText"]
                for entities in line_json["entities"]:
                    entity_list.append({"entity_index": {"begin": entities["start_pos"],
                                                         "end":  entities["end_pos"]},
                                                "entity_type": entities["label_type"],
                                        "entity": text[entities["start_pos"]:entities["end_pos"]]})
                g.write(json.dumps({"text": text, "entity_list": entity_list}, ensure_ascii=False) + "\n")
                length += 1
            except:
                error += 1
        print("错误：{}个".format(error))
        print("共有{}行".format(length))


# 统计实体类型和个数
def sta_entity(file, num=None):
    sta_dict = defaultdict(int)
    with open(file, encoding="utf-8") as f:
        data_list = list(f.readlines())

        length = len(data_list) if not num else len

        entity_type = []
        for line in data_list[:length]:
            text_e = json.loads(line)
            for e in text_e["entity_list"]:
                if e["entity_type"] not in entity_type:
                    entity_type.append(e["entity_type"])
                sta_dict[e["entity_type"]] += 1

        entity_type.sort()
        print("实体类型：",entity_type)
        print("实体类型及个数：", sta_dict)


print("train1")
# transfer_data_5("yidu-s4k/subtask1_training_part1.txt", "yidu-s4k/train1.txt")
sta_entity("yidu-s4k/train1.txt")
print("train2")
transfer_data_5("yidu-s4k/subtask1_training_part2.txt", "yidu-s4k/train2.txt")
sta_entity("yidu-s4k/train2.txt")
print("test")
transfer_data_5("yidu-s4k/subtask1_test_set_with_answer.json", "yidu-s4k/test.txt")
sta_entity("yidu-s4k/test.txt")