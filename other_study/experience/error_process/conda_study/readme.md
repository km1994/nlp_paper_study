# conda 操作

## 创建新环境

使用 

```s
conda create -n your_env_name python=X.X（2.7、3.6等)
```

命令创建python版本为X.X、名字为your_env_name的虚拟环境。your_env_name文件可以在Anaconda安装目录envs文件下找到。

## jupyter 添加新环境

```s
conda install -n 环境名称 ipykernel
activate 环境名称
python -m ipykernel install --user --name 环境名称 --display-name "在jupyter中显示的环境名称"
```

eg：
```s
conda install -n checklist ipykernel
activate checklist
python -m ipykernel install --user --name checklist --display-name "在jupyter中显示的环境名称"
```

## 加载 en_core_web_sm 语料库

```s
conda install spacy
```
- 加载数据 https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
- 
- 安装
```
pip install en_core_web_sm-2.0.0.tar.gz
```


