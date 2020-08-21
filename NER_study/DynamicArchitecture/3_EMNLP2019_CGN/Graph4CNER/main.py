from utils.data import Data
from utils.batchify import batchify
from utils.config import get_args
from utils.metric import get_ner_fmeasure
from model.bilstm_gat_crf import BLSTM_GAT_CRF
import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
import time
import random
import sys
import gc


def data_initialization(args):
    data_stored_directory = args.data_stored_directory
    file = data_stored_directory + args.dataset_name + "_dataset.dset"
    if os.path.exists(file) and not args.refresh:
        data = load_data_setting(data_stored_directory, args.dataset_name)
    else:
        data = Data()
        data.dataset_name = args.dataset_name
        data.norm_char_emb = args.norm_char_emb
        data.norm_gaz_emb = args.norm_gaz_emb
        data.number_normalized = args.number_normalized
        data.max_sentence_length = args.max_sentence_length
        data.build_gaz_file(args.gaz_file)
        data.generate_instance(args.train_file, "train", False)
        data.generate_instance(args.dev_file, "dev")
        data.generate_instance(args.test_file, "test")
        data.build_char_pretrain_emb(args.char_embedding_path)
        data.build_gaz_pretrain_emb(args.gaz_file)
        data.fix_alphabet()
        data.get_tag_scheme()
        save_data_setting(data, data_stored_directory)
    return data


def save_data_setting(data, data_stored_directory):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(data_stored_directory):
        os.makedirs(data_stored_directory)
    dataset_saved_name = data_stored_directory + data.dataset_name +"_dataset.dset"
    with open(dataset_saved_name, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", dataset_saved_name)


def load_data_setting(data_stored_directory, name):
    dataset_saved_name = data_stored_directory + name + "_dataset.dset"
    with open(dataset_saved_name, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", dataset_saved_name)
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def evaluate(data, model, args, name):
    if name == "train":
        instances = data.train_ids
    elif name == "dev":
        instances = data.dev_ids
    elif name == 'test':
        instances = data.test_ids
    else:
        print("Error: wrong evaluate name,", name)
    pred_results = []
    gold_results = []
    model.eval()
    batch_size = args.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        char, c_len, gazs, mask, label, recover, t_graph, c_graph, l_graph = batchify(instance, args.use_gpu)
        tag_seq = model(char, c_len, gazs, t_graph, c_graph, l_graph, mask)
        pred_label, gold_label = recover_label(tag_seq, label, mask, data.label_alphabet, recover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagscheme)
    return speed, acc, p, r, f, pred_results


def train(data, model, args):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "Adam":
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    best_dev = -1
    for idx in range(args.max_epoch):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, args.max_epoch))
        optimizer = lr_decay(optimizer, idx, args.lr_decay, args.lr)
        instance_count = 0
        sample_loss = 0
        total_loss = 0
        random.shuffle(data.train_ids)
        model.train()
        model.zero_grad()
        batch_size = args.batch_size
        train_num = len(data.train_ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > train_num:
                end = train_num
            instance = data.train_ids[start:end]
            if not instance:
                continue
            model.zero_grad()
            char, c_len, gazs, mask, label, recover, t_graph, c_graph, l_graph = batchify(instance, args.use_gpu)
            loss = model.neg_log_likelihood(char, c_len, gazs, t_graph, c_graph, l_graph, mask, label)
            instance_count += 1
            sample_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            if args.use_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            model.zero_grad()
            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f" % (
                end, temp_cost, sample_loss))
                sys.stdout.flush()
                sample_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f" % (end, temp_cost, sample_loss))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        speed, acc, p, r, f, _ = evaluate(data, model, args, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        current_score = f
        print(
            "Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))
        if current_score > best_dev:
            print("Exceed previous best f score:", best_dev)
            if not os.path.exists(args.param_stored_directory + args.dataset_name + "_param"):
                os.makedirs(args.param_stored_directory + args.dataset_name + "_param")
            model_name = "{}epoch_{}_f1_{}.model".format(args.param_stored_directory + args.dataset_name + "_param/", idx, current_score)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
        gc.collect()


if __name__ == '__main__':
    args, unparsed = get_args()
    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False   # True
    data = data_initialization(args)
    model = BLSTM_GAT_CRF(data, args)
    train(data, model, args)


