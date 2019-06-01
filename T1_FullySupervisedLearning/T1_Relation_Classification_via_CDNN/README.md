## Relation Classification via Convolutional Deep Neural Network

url : https://github.com/FrankWork/conv_relation
TensorFlow implementation of [the paper](http://www.aclweb.org/anthology/C14-1220),

dataset: SemEval2010 task8

word embeddings: senna

to run the code:
```
./run
```

## Environment(have tested)
- tensorflow 1.4.0
- python 3.5
- linux,macOs or windows

## How to run ?

- to train model

    `./run`
    
    where ```num_epochs=200 --word_dim=50```have been set in 'run' file.
-  to test model
 
    excute 

    `python src/train.py  --num_epochs=200 --word_dim=50 --test`

    then you can get a 'results.txt'  file in ```/data/resuts.txt```

- to calculate F1 score

    ```perl src/scorer.pl data/test_keys.txt data/results.txt```



##  Problem
when you use Spyder or PyCharm to run this code, you may encounter this error: 

```
ArgumentError: argument --train_file: conflicting option string: --train_file
```

solution:

1. restart spyder

2. or add annotation for all definitions of ```tf.flags.FLAGS``` .

such as ```# flags.DEFINE_string("train_file", "data/train.cln", 
                             "original training file")```

## Difference 

1. delete 'the hidden layer 2' as [the paper mentioned](http://www.aclweb.org/anthology/C14-1220)
2. use muti - window size(w=3,w=4,w=5) in convolution layer
3. delete Wordnet lexical feature




