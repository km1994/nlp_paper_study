from pytorch_transformers import BertConfig
config = BertConfig.from_pretrained("F:/document/datasets/nlpData/pretrain/bert/chinese_wwm_ext_pytorch")

print(f"config.hidden_size:{config.hidden_size}")

