# LGN

Pytorch implementation of [A Lexicon-Based Graph Neural Network for Chinese NER](https://www.aclweb.org/anthology/D19-1096.pdf).

The code is partially referred to https://github.com/jiesutd/LatticeLSTM.

## Requirements

* Python 3.6 or higher
* Pytorch 0.4.1 or higher

## Input Format

BMES tag scheme, with each character its label for one line. Sentences are splited with a null line.

	印   B-LOC
	度   M-LOC
	河   E-LOC
	流   O
	经   O
	印   B-GPE
	度   E-GPE

## Usage

* Training

		python main.py --status train \
		               --train data/onto4ner.cn/train.char.bmes \
		               --dev data/onto4ner.cn/dev.char.bmes \
		               --test data/onto4ner.cn/test.char.bmes \
		               --saved_model saved_model/model_onto4ner \
		               --saved_set data/onto4ner.cn/saved.dset

		python main.py --status train  --train data/weiboNER.conll.train  --dev data/weiboNER.conll.dev --test data/weiboNER.conll.test   --saved_model saved_model/model_weiboNER   --saved_set data/saved.dset
		               
* Testing

		python main.py --status test \
		               --test data/onto4ner.cn/test.char.bmes \
		               --saved_model saved_model/model_onto4ner \
		               --saved_set data/onto4ner.cn/saved.dset
		               
* Decoding (Raw file can either be labeled or not.)

		python main.py --status decode \
		               --raw data/onto4ner.cn/test.char.bmes \
		               --output tagged_file.txt \
		               --saved_model saved_model/model_onto4ner \
		               --saved_set data/onto4ner.cn/saved.dset
		               
## Data

The pretrained character and word embeddings can be downloaded from [Lattice LSTM](https://github.com/jiesutd/LatticeLSTM).

Original datasets can be found at [OntoNotes](https://catalog.ldc.upenn.edu/LDC2011T03), [MSRA](http://sighan.cs.uchicago.edu/bakeoff2006/), 
[Weibo](https://github.com/hltcoe/golden-horse) and [Resume](https://github.com/jiesutd/LatticeLSTM/tree/master/ResumeNER).
If you want preprocessed datasets that satisfy the input format of our codes, please contact me.

## Pretrained Model Downloads

We also provide pretrained models on the four datasets, which are the same models as reported in the paper.
If you try to retrain models from scratch under the same hyper-parameter settings, you may obtain a sightly 
lower or higher F1 score than that reported in the paper (in our experiments we selected the model that performed best).

Pretrained models and related hyper-parameter settings are available at [Google Drive](https://drive.google.com/file/d/1KKkCW8WRhgR2P2UbRpNpKyE_RAv1EREv/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1U89EwnhPpMa4bNrS--u4EA).

When running main.py in test mode for pretrained models, you can get the results as follows:

| Datasets       | Precision | Recall  | F1    | 
|:--------------:|:---------:|:-------:|:-----:|
| OntoNotes dev  |   74.00   |  70.03  | 71.96 |
| OntoNotes test |   76.13   |  73.68  | 74.89 | 
| MSRA dev       |     -     |   -     |   -   |
| MSRA test      |   94.19   |  92.73  | 93.46 |
| Weibo dev      |   66.09   |  59.13  | 62.42 |
| Weibo test     |   65.71   |  55.56  | 60.21 |
| Resume dev     |   94.27   |  94.59  | 94.43 |
| Resume test    |   95.28   |  95.46  | 95.37 |

## Citation

	@inproceedings{gui2019lexicon,
  	 title={A Lexicon-Based Graph Neural Network for Chinese NER},
  	 author={Gui, Tao and Zou, Yicheng and Zhang, Qi and Peng, Minlong and 
	 Fu, Jinlan and Wei, Zhongyu and Huang, Xuanjing},
  	 booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing 
	 and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  	 pages={1039--1049},
  	 year={2019}
	}
