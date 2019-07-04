# Relation Classification via Attention Model
As you know the attention model can help us to solve many problems.Resently, I have a project which need to recognize the relation from some entities. After reading several paper, I decided to implement this paper: [Relation Classification via Multi-Level Attention CNNs](http://iiis.tsinghua.edu.cn/~weblt/papers/relation-classification.pdf)
I desperately desire to use pytorch to do some awsome things. So it's the only choice for me. And i think you will like it.

Some of data handling codes are copied from [ACNN](https://github.com/FrankWork/acnn)

You need an environment:
pytorch 1.0.0
keras & tensorflow (I only used one function which name is to_categorical)
Git this project to your pycharm or other IDE, then edit the acnn_train.py to satisfied your data
# 18.12.17 The Renewed Version
These days, I reviewed the paper again and update my code. But the acc is still low.

Could somebody can give me some advice?
# Network Structure
<p align="center"><img width="60%" src="acnn_structure.png" /></p>
