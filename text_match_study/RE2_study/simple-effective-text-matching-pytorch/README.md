# RE2

This is a pytorch implementation of the ACL 2019 paper "Simple and Effective Text Matching with Richer Alignment Features". The original Tensorflow implementation: https://github.com/alibaba-edu/simple-effective-text-matching.

## Quick Links

- [About](#simple-and-effective-text-matching)
- [Setup](#setup)
- [Usage](#usage)

## Simple and Effective Text Matching

RE2 is a fast and strong neural architecture for general purpose text matching applications. 
In a text matching task, a model takes two text sequences as input and predicts their relationship.
This method aims to explore what is sufficient for strong performance in these tasks. 
It simplifies many slow components which are previously considered as core building blocks in text matching,
while keeping three key features directly available for inter-sequence alignment: 
original point-wise features, previous aligned features, and contextual features.

RE2 achieves performance on par with the state of the art on four benchmark datasets: SNLI, SciTail, Quora and WikiQA,
across tasks of natural language inference, paraphrase identification and answer selection 
with no or few task-specific adaptations. It has at least 6 times faster inference speed compared to similarly performed models.

<p align="center"><img width="50%" src="figure.png" /></p>

The following table lists major experiment results. 
The paper reports the average and standard deviation of 10 runs. 
Inference time (in seconds) is measured by processing a batch of 8 pairs of length 20 on Intel i7 CPUs.
The computation time of POS features used by CSRAN and DIIN is not included.

|Model|SNLI|SciTail|Quora|WikiQA|Inference Time|
|---|---|---|---|---|---|
|[BiMPM](https://github.com/zhiguowang/BiMPM)|86.9|-|88.2|0.731|0.05|
|[ESIM](https://github.com/lukecq1231/nli)|88.0|70.6|-|-|-|
|[DIIN](https://github.com/YichenGong/Densely-Interactive-Inference-Network)|88.0|-|89.1|-|1.79|
|[CSRAN](https://github.com/vanzytay/EMNLP2018_NLI)|88.7|86.7|89.2|-|0.28|
|RE2|88.9±0.1|86.0±0.6|89.2±0.2|0.7618 ±0.0040|0.03~0.05|

Refer to the paper for more details of the components and experiment results.

## Setup

- install python >= 3.6 and pip
- `pip install -r requirements.txt`
- install [PyTorch](https://pytorch.org) (1.0 <= version < 1.2). PyTorch 1.2 has breaking changes on masking and will be supported in the future. Some useful links for pip wheels:
    - [linux/pytorch 1.0.1/CUDA 8/python 3.6](https://download.pytorch.org/whl/cu80/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl)
    - [linux/pytorch 1.0.1/CUDA 8/python 3.7](https://download.pytorch.org/whl/cu80/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl)
    - [linux/pytorch 1.0.1/CUDA 9/python 3.6](https://download.pytorch.org/whl/cu90/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl)
    - [linux/pytorch 1.0.1/CUDA 9/python 3.7](https://download.pytorch.org/whl/cu90/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl)
- Download [GloVe word vectors](https://nlp.stanford.edu/projects/glove/) (glove.840B.300d) to `resources/`

Data used in the paper are prepared as follows:

### SNLI

- Download and unzip [SNLI](https://www.dropbox.com/s/0r82spk628ksz70/SNLI.zip?dl=0) 
(pre-processed by [Tay et al.](https://github.com/vanzytay/EMNLP2018_NLI)) to `data/orig`. 
- Unzip all zip files in the "data/orig/SNLI" folder. (`cd data/orig/SNLI && gunzip *.gz`)
- `cd data && python prepare_snli.py` 

### SciTail

- Download and unzip [SciTail](http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip) 
dataset to `data/orig`.
- `cd data && python prepare_scitail.py`

### Quora

- Download and unzip [Quora](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing)
dataset (pre-processed by [Wang et al.](https://github.com/zhiguowang/BiMPM)) to `data/orig`.
- `cd data && python prepare_quora.py`

### WikiQA

- Download and unzip [WikiQA](https://www.microsoft.com/en-us/download/details.aspx?id=52419)
to `data/orig`.
- `cd data && python prepare_wikiqa.py`
- Download and unzip [evaluation scripts](http://cs.stanford.edu/people/mengqiu/data/qg-emnlp07-data.tgz). 
Use the `make -B` command to compile the source files in `qg-emnlp07-data/eval/trec_eval-8.0`.
Move the binary file "trec_eval" to `resources/`.

## Usage

To train a new text matching model, run the following command: 

```bash
python train.py $config_file.json5
```

Example configuration files are provided in `configs/`:

- `configs/main.json5`: replicate the main experiment result in the paper.
- `configs/robustness.json5`: robustness checks
- `configs/ablation.json5`: ablation study

The instructions to write your own configuration files:

```json5
[
    {
        name: 'exp1', // name of your experiment, can be the same across different data
        __parents__: [
            'default', // always put the default on top
            'data/quora', // data specific configurations in `configs/data`
            // 'debug', // use "debug" to quick debug your code  
        ],
        __repeat__: 5,  // how may repetitions you want
        blocks: 3, // other configurations for this experiment 
    },
    // multiple configurations are executed sequentially
    {
        name: 'exp2', // results under the same name will be overwritten
        __parents__: [
            'default', 
            'data/quora',
        ],
        __repeat__: 5,  
        blocks: 4, 
    }
]
```

To check the configurations only, use

```bash
python train.py $config_file.json5 --dry
```

To evaluate an existed model, use `python evaluate.py $model_path $data_file`, here's an example:

```bash
python evaluate.py models/snli/benchmark/best.pt data/snli/train.txt 
python evaluate.py models/snli/benchmark/best.pt data/snli/test.txt 
```

> Note that multi-GPU training is not yet supported in the pytorch implementation. A single 16G GPU is sufficient for training when blocks < 5 with hidden size 200 and batch size 512. All the results reported in the paper except the robustness checks can be reproduced with a single 16G GPU.

## Citation

Please cite the ACL paper if you use RE2 in your work:

```
@inproceedings{yang2019simple,
  title={Simple and Effective Text Matching with Richer Alignment Features},
  author={Yang, Runqi and Zhang, Jianhai and Gao, Xing and Ji, Feng and Chen, Haiqing},
  booktitle={Association for Computational Linguistics (ACL)},
  year={2019}
}
```

## License
This project is under Apache License 2.0.
