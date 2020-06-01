# [ACL'19 (long)] GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction
A **PyTorch** implementation of GraphRel

[Project](https://tsujuifu.github.io/projs/acl19_graph-rel.html) | [Paper](https://tsujuifu.github.io/pubs/acl19_graph-rel.pdf) | [Poster](https://github.com/tsujuifu/pytorch_graph-rel/raw/master/imgs/poster.png)

<img src='imgs/result.png' width='85%' />

## Overview
GraphRel is an implementation of <br> 
"[GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction](https://tsujuifu.github.io/pubs/acl19_graph-rel.pdf)" <br>
[Tsu-Jui Fu](http://tsujuifu.github.io/), [Peng-Hsuan Li](http://jacobvsdanniel.github.io/), and [Wei-Yun Ma](http://www.iis.sinica.edu.tw/pages/ma/) <br>
in Annual Meeting of the Association for Computational Linguistics (**ACL**) 2019 (long)

<img src='imgs/overview.png' width='80%' />

In the 1st-phase, we **adopt bi-RNN and GCN to extract both sequential and regional dependency** word features. Given the word features, we **predict relations for each word pair** and the entities for all words. Then, in 2nd-phase, based on the predicted 1st-phase relations, we build complete relational graphs for each relation, to which we **apply GCN on each graph to integrate each relation’s information** and further consider the interaction between entities and relations.

## Requirements
This code is implemented under **Python3** and [PyTorch](https://pytorch.org). <br>
Following libraries are also required:
+ [PyTorch](https://pytorch.org) >= 0.4
+ [spaCy](https://spacy.io)

## Usage
We use [spaCy](https://spacy.io/) as **pre-trained word embedding** and **dependency parser**.

+ GraphRel
```
model_graph-rel.ipynb
```

## Resources
+ NYT Dataset
+ WebNLG Dataset
+ [This project](https://drive.google.com/drive/folders/1BvqVpGX7gfZLUXN3AxxCs3Ik618IWz-L?usp=sharing)

## Citation
```
@inproceedings{fu2019graph-rel, 
  author = {Tsu-Jui Fu and Peng-Hsuan Li and Wei-Yun Ma}, 
  title = {GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extractionn}, 
  booktitle = {Annual Meeting of the Association for Computational Linguistics (ACL)}, 
  year = {2019} 
}
```

## Acknowledgement
+ [copy_re](https://github.com/xiangrongzeng/copy_re)
