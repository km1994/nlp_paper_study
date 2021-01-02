# Named Entity Recognition as Dependency Parsing

## Introduction
This repository contains code introduced in the following paper:
 
**[Named Entity Recognition as Dependency Parsing](https://www.aclweb.org/anthology/2020.acl-main.577/)**  
Juntao Yu, Bernd Bohnet and Massimo Poesio  
In *Proceedings of the 58th Annual Conference of the Association for Computational Linguistics (ACL)*, 2020

## Setup Environments
* The code is written in Python 2 and Tensorflow 1.0, A Python3 and Tensorflow 2.0 version is provided by Amir (see **Other Versions**).  
* Before starting, you need to install all the required packages listed in the requirment.txt using `pip install -r requirements.txt`.
* Then download the BERT models, for English we used the [original cased BERT-Large model](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip) and for other languages we used the [cased BERT-Base multilingual model]( https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip).
* After that modify and run `extract_bert_features/extract_bert_features.sh` to compute the BERT embeddings for your training or testing.
* You also need to download context-independent word embeddings such as fasttext or GloVe embeddings that required by the system.

## To use a pre-trained model
* Pre-trained models can be download from [this link](https://www.dropbox.com/s/vx30kijnvio1f4k/acl2020%20best%20models.zip?dl=0). We provide all nine pre-trained models reported in our paper.
* Choose the model you want to use and copy them to the `logs/` folder.
* Modifiy the *test_path* accordingly in the `experiments.conf`:
   * the *test_path* is the path to *.jsonlines* file, each line of the *.jsonlines* file is a batch of sentences and must in the following format:
   
   ```
  {"doc_key": "batch_01", 
  "ners": [[[0, 0, "PER"], [3, 3, "GPE"], [5, 5, "GPE"]], 
  [[3, 3, "PER"], [10, 14, "ORG"], [20, 20, "GPE"], [20, 25, "GPE"], [22, 22, "GPE"]], 
  []], 
  "sentences": [["Anwar", "arrived", "in", "Shanghai", "from", "Nanjing", "yesterday", "afternoon", "."], 
  ["This", "morning", ",", "Anwar", "attended", "the", "foundation", "laying", "ceremony", "of", "the", "Minhang", "China-Malaysia", "joint-venture", "enterprise", ",", "and", "after", "that", "toured", "Pudong", "'s", "Jingqiao", "export", "processing", "district", "."], 
  ["(", "End", ")"]]}
  ```
  
  * Each of the sentences in the batch corresponds to a list of NEs stored under `ners` key, if some sentences do not contain NEs use an empty list `[]` instead.
* Then use `python evaluate.py config_name` to start your evaluation

## To train your own model
* You will need additionally to create the character vocabulary by using `python get_char_vocab.py train.jsonlines dev.jsonlines`
* Then you can start training by using `python train.py config_name`

## Other Versions
* [Amir Zeldes](https://github.com/amir-zeldes) kindly created a tensorflow 2.0 and python 3 ready version and can be find [here](https://github.com/amir-zeldes/biaffine-ner)
