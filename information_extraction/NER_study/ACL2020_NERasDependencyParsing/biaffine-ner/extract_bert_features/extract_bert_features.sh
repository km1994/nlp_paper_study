export BERT_MODEL_PATH="./cased_L-24_H-1024_A-16"
PYTHONPATH=. python extract_features.py --input_file="train.jsonlines;dev.jsonlines;test.jsonlines" --output_file=../bert_features.hdf5 --bert_config_file $BERT_MODEL_PATH/bert_config.json --init_checkpoint $BERT_MODEL_PATH/bert_model.ckpt --vocab_file  $BERT_MODEL_PATH/vocab.txt --do_lower_case=False --stride 1 --window_size 511
