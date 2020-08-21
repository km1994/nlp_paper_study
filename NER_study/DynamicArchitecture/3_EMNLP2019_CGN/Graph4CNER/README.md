# Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network
[![GitHub stars](https://img.shields.io/github/stars/DianboWork/Graph4CNER?style=flat-square)](https://github.com/DianboWork/Graph4CNER/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/DianboWork/Graph4CNER?style=flat-square&color=blueviolet)](https://github.com/DianboWork/Graph4CNER/network/members)

Source code for [Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network](https://www.aclweb.org/anthology/D19-1396.pdf) in EMNLP 2019. If you use this code or our results in your research, we would appreciate it if you cite our paper as following:


```
@article{Sui2019Graph4CNER,
    title = {Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network},
    author = {Sui, Dianbo and Chen, Yubo and Liu, Kang and Zhao, Jun and Liu, Shengping},
    journal = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
    year = {2019}
}
```
Requirements:
======
	Python: 3.7   
	PyTorch: 1.1.0 

Input format:
======
Input is in CoNLL format (We use BIO tag scheme), where each character and its label are in one line. Sentences are split with a null line.

	叶 B-PER
	嘉 I-PER
	莹 I-PER
	先 O
	生 O
	获 O
	聘 O
	南 B-ORG
	开 I-ORG
	大 I-ORG
	学 I-ORG
	终 O
	身 O
	校 O
	董 O
	。 O

Pretrained Embeddings:
====
Character embeddings (gigaword_chn.all.a2b.uni.ite50.vec) can be downloaded in [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D).

Word embeddings (sgns.merge.word) can be downloaded in [Google Drive](https://drive.google.com/file/d/1Zh9ZCEu8_eSQ-qkYVQufQDNKPC4mtEKR/view) or
[Baidu Pan](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw).

Usage：
====
:one: Download the character embeddings and word embeddings and put them in the `data/embeddings` folder.

:two: Modify the `run_main.sh` by adding your train/dev/test file directory.

:three: `sh run_main.sh`. Note that the default hyperparameters is may not be the optimal hyperparameters, and you need to adjust these.

:four: Enjoy it! :smile:

Result：
====
For WeiboNER dataset, using the default hyperparameters in `run_main.sh` can achieve the state-of-art results (Test F1: 66.66%). Model parameters can be download in [Baidu Pan](https://pan.baidu.com/s/1ysy_eNF0oYJwjXiy4x7gtQ) (key: bg3q):sunglasses:

Speed：
===
I have optimized the code and this version is faster than the one in our paper. :muscle:
