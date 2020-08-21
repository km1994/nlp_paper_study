import argparse
parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg

net_arg = add_argument_group('Network')
net_arg.add_argument('--fix_gaz_emb', type=str2bool, default=True)
net_arg.add_argument('--lstm_layer', type=int, default=1)
net_arg.add_argument('--bilstm_flag', type=str2bool, default=True)
net_arg.add_argument('--gat_nhidden', type=int, default=30)
net_arg.add_argument('--gat_nhead', type=int, default=3)
net_arg.add_argument('--gat_layer', type=int, default=2, choices=[1, 2])
net_arg.add_argument('--strategy', type=str, default="m", choices=['v', 'n', 'm'])
net_arg.add_argument("--alpha", type=float, default=0.1)
net_arg.add_argument('--dropout', type=float, default=0.5)
net_arg.add_argument('--droplstm', type=float, default=0.5)
net_arg.add_argument('--dropgat', type=float, default=0.5)
net_arg.add_argument('--gaz_dropout', type=float, default=0.5)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset_name', type=str, default='Data')
data_arg.add_argument('--train_file', type=str, help="train file")
data_arg.add_argument('--test_file', type=str, help="test file")
data_arg.add_argument('--dev_file', type=str, help="dev file")
data_arg.add_argument('--gaz_file', type=str, default="D:/resource/Graph4CNER/sgns.merge.word", help="lexical embeddings file")
data_arg.add_argument('--char_embedding_path', type=str, default="D:/resource/Graph4CNER/gigaword_chn.all.a2b.uni.ite50.vec",help="characher embeddings file")
data_arg.add_argument('--data_stored_directory', type=str, default="./data/generated_data/")
data_arg.add_argument('--param_stored_directory', type=str, default="./data/model_param/")

preprocess_arg = add_argument_group('Preprocess')
preprocess_arg.add_argument('--norm_char_emb', type=str2bool, default=False)
preprocess_arg.add_argument('--norm_gaz_emb', type=str2bool, default=True)
preprocess_arg.add_argument('--number_normalized', type=str2bool, default=True)
preprocess_arg.add_argument('--max_sentence_length', type=int, default=250)

learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--batch_size', type=int, default=20)
learn_arg.add_argument('--max_epoch', type=int, default=150)
learn_arg.add_argument('--lr', type=float, default=0.001)
learn_arg.add_argument('--lr_decay', type=float, default=0.01)
learn_arg.add_argument('--use_clip', type=str2bool, default=False)
learn_arg.add_argument('--clip', type=float, default=5.0)
learn_arg.add_argument("--optimizer", type=str, default="Adam", choices=['Adam', 'SGD'])
learn_arg.add_argument("--l2_penalty", type=float, default=0.00000005)
# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--refresh', type=str2bool, default=False)
misc_arg.add_argument('--use_gpu', type=str2bool, default=False)
# misc_arg.add_argument('--visible_gpu', type=int, default=0)
misc_arg.add_argument('--random_seed', type=int, default=1)


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed
