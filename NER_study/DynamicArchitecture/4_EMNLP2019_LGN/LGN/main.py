# -*- coding: utf-8 -*-
# @Author: Yicheng Zou
# @Last Modified by:   Yicheng Zou,    Contact: yczou18@fudan.edu.cn

import time
import sys
import argparse
import random
import torch
import gc
import pickle
import os
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.LGN import Graph
from utils.data import Data


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print( " Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'aggr':
            param_group['lr'] = lr * 2.
        else:
            param_group['lr'] = lr
    return optimizer


def data_initialization(data, word_file, train_file, dev_file, test_file):

    data.build_word_file(word_file)

    if train_file:
        data.build_alphabet(train_file)
        data.build_word_alphabet(train_file)
    if dev_file:
        data.build_alphabet(dev_file)
        data.build_word_alphabet(dev_file)
    if test_file:
        data.build_alphabet(test_file)
        data.build_word_alphabet(test_file)
    return data


def predict_check(pred_variable, gold_variable, mask_variable):

    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet):

    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    pred_label = []
    gold_label = []

    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)

    return pred_label, gold_label


def print_args(args):
    print("CONFIG SUMMARY:")
    print("     Batch size: %s" % (args.batch_size))
    print("     If use GPU: %s" % (args.use_gpu))
    print("     If use CRF: %s" % (args.use_crf))
    print("     Epoch  number: %s" % (args.num_epoch))
    print("     Learning rate: %s" % (args.lr))
    print("     L2 normalization rate: %s" % (args.weight_decay))
    print("     If use edge embedding: %s" % (args.use_edge))
    print("     If  use  global  node: %s" % (args.use_global))
    print("     Bidirectional digraph: %s" % (args.bidirectional))
    print("     Update   step  number: %s" % (args.iters))
    print("     Attention  dropout   rate: %s" % (args.tf_drop_rate))
    print("     Embedding  dropout   rate: %s" % (args.emb_drop_rate))
    print("     Hidden  state   dimension: %s" % (args.hidden_dim))
    print("     Learning rate decay ratio: %s" % (args.lr_decay))
    print("     Aggregation module dropout rate: %s" % (args.cell_drop_rate))
    print("     Head    number   of   attention: %s" % (args.num_head))
    print("     Head  dimension   of  attention: %s" % (args.head_dim))
    print("CONFIG SUMMARY END.")
    sys.stdout.flush()


def evaluate(data, args, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(0)

    pred_results = []
    gold_results = []

    # set model in eval model
    model.eval()
    batch_size = args.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1

    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue

        word_list, batch_char, batch_label, mask = batchify_with_label(instance, args.use_gpu)
        _, tag_seq = model(word_list, batch_char, mask)

        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet)

        pred_results += pred_label
        gold_results += gold_label

    decode_time = time.time() - start_time
    speed = len(instances) / decode_time

    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results)
    return speed, acc, p, r, f, pred_results


def batchify_with_label(input_batch_list, gpu):

    batch_size = len(input_batch_list)
    chars = [sent[0] for sent in input_batch_list]
    words = [sent[1] for sent in input_batch_list]
    labels = [sent[2] for sent in input_batch_list]

    sent_lengths = torch.LongTensor(list(map(len, chars)))
    max_sent_len = sent_lengths.max()
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_sent_len))).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_sent_len))).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_sent_len))).byte()

    for idx, (seq, label, seq_len) in enumerate(zip(chars, labels, sent_lengths)):
        char_seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seq_len] = torch.LongTensor(label)
        mask[idx, :seq_len] = torch.Tensor([1] * int(seq_len))

    if gpu:
        char_seq_tensor = char_seq_tensor.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()

    return words, char_seq_tensor, label_seq_tensor, mask


def train(data, args, saved_model_path):

    print( "Training model...")
    model = Graph(data, args)
    if args.use_gpu:
        model = model.cuda()
    print('# generated parameters:', sum(param.numel() for param in model.parameters()))
    print( "Finished built model.")

    best_dev_epoch = 0
    best_dev_f = -1
    best_dev_p = -1
    best_dev_r = -1

    best_test_f = -1
    best_test_p = -1
    best_test_r = -1

    # Initialize the optimizer
    aggr_module_params = []
    other_module_params = []
    for m_name in model._modules:
        m = model._modules[m_name]
        if isinstance(m, torch.nn.ModuleList):
            for p in m.parameters():
                if p.requires_grad:
                    aggr_module_params.append(p)
        else:
            for p in m.parameters():
                if p.requires_grad:
                    other_module_params.append(p)

    optimizer = optim.Adam([
            {"params": (aggr_module_params), "name": "aggr"},
            {"params": (other_module_params), "name": "other"}
        ],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    for idx in range(args.num_epoch):
        epoch_start = time.time()
        temp_start = epoch_start
        print(("Epoch: %s/%s" %(idx, args.num_epoch)))
        optimizer = lr_decay(optimizer, idx, args.lr_decay, args.lr)
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        # set model in train model
        model.train()
        model.zero_grad()
        batch_size = args.batch_size
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1

        for batch_id in range(total_batch):
            # Get one batch-sized instance
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue

            word_list, batch_char, batch_label, mask = batchify_with_label(instance, args.use_gpu)
            loss, tag_seq = model(word_list, batch_char, mask, batch_label)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data
            total_loss += loss.data
            batch_loss += loss

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" %
                       (end, temp_cost, sample_loss, right_token, whole_token, (right_token+0.)/whole_token)))
                sys.stdout.flush()
                sample_loss = 0
            if end % args.batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0

        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" %
               (end, temp_cost, sample_loss, right_token, whole_token, (right_token+0.)/whole_token)))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print(("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" %
               (idx, epoch_cost, train_num/epoch_cost, total_loss)))

        # dev
        speed, acc, dev_p, dev_r, dev_f, _ = evaluate(data, args, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        print(("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" %
               (dev_cost, speed, acc, dev_p, dev_r, dev_f)))

        # test
        speed, acc, test_p, test_r, test_f, _ = evaluate(data, args, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish

        print(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" %
               (test_cost, speed, acc, test_p, test_r, test_f)))

        if dev_f > best_dev_f:
            print("Exceed previous best f score: %.4f" % best_dev_f)
            torch.save(model.state_dict(), saved_model_path + "_best")
            best_dev_p = dev_p
            best_dev_r = dev_r
            best_dev_f = dev_f
            best_dev_epoch = idx + 1
            best_test_p = test_p
            best_test_r = test_r
            best_test_f = test_f

        model_idx_path = saved_model_path + "_" + str(idx)
        torch.save(model.state_dict(), model_idx_path)
        with open(saved_model_path + "_result.txt", "a") as file:
            file.write(model_idx_path + '\n')
            file.write("Dev score: %.4f, r: %.4f, f: %.4f\n" % (dev_p, dev_r, dev_f))
            file.write("Test score: %.4f, r: %.4f, f: %.4f\n\n" % (test_p, test_r, test_f))
            file.close()

        print("Best dev epoch: %d" % best_dev_epoch)
        print("Best dev score: p: %.4f, r: %.4f, f: %.4f" % (best_dev_p, best_dev_r, best_dev_f))
        print("Best test score: p: %.4f, r: %.4f, f: %.4f" % (best_test_p, best_test_r, best_test_f))

        gc.collect()

    with open(saved_model_path + "_result.txt", "a") as file:
        file.write("Best epoch: %d" % best_dev_epoch + '\n')
        file.write("Best Dev score: %.4f, r: %.4f, f: %.4f\n" % (best_dev_p, best_dev_r, best_dev_f))
        file.write("Test score: %.4f, r: %.4f, f: %.4f\n\n" % (best_test_p, best_test_r, best_test_f))
        file.close()

    with open(saved_model_path + "_best_HP.config", "wb") as file:
        pickle.dump(args, file)


def load_model_decode(model_dir, data, args, name):
    model_dir = model_dir + "_best"
    print("Load Model from file: ", model_dir)
    model = Graph(data, args)
    model.load_state_dict(torch.load(model_dir))

    # load model need consider if the model trained in GPU and load in CPU, or vice versa
    if args.use_gpu:
        model = model.cuda()

    print(("Decode %s data ..." % name))
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, args, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" %
           (name, time_cost, speed, acc, p, r, f)))

    return pred_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='Function status.', default='train')
    parser.add_argument('--use_gpu', type=str2bool, default=False)
    parser.add_argument('--train', help='Training set.', default='data/onto4ner.cn/train.char.bmes')
    parser.add_argument('--dev', help='Developing set.', default='data/onto4ner.cn/dev.char.bmes')
    parser.add_argument('--test', help='Testing set.', default='data/onto4ner.cn/test.char.bmes')
    parser.add_argument('--raw', help='Raw file for decoding.')
    parser.add_argument('--output', help='Output results for decoding.')
    parser.add_argument('--saved_set', help='Path of saved data set.', default='data/onto4ner.cn/saved.dset')
    parser.add_argument('--saved_model', help='Path of saved model.', default="saved_model/model_onto4ner")
    parser.add_argument('--char_emb', help='Path of character embedding file.', default="data/gigaword_chn.all.a2b.uni.ite50.vec")
    parser.add_argument('--word_emb', help='Path of word embedding file.', default="data/ctb.50d.vec")

    parser.add_argument('--use_crf', type=str2bool, default=True)
    parser.add_argument('--use_edge', type=str2bool, default=True, help='If use lexicon embeddings (edge embeddings).')
    parser.add_argument('--use_global', type=str2bool, default=True, help='If use the global node.')
    parser.add_argument('--bidirectional', type=str2bool, default=True, help='If use bidirectional digraph.')

    parser.add_argument('--seed', help='Random seed', default=1023, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=1, type=int)
    parser.add_argument('--num_epoch',default=100, type=int, help="Epoch number.")
    parser.add_argument('--iters', default=4, type=int, help='The number of Graph iterations.')
    parser.add_argument('--hidden_dim', default=50, type=int, help='Hidden state size.')
    parser.add_argument('--num_head', default=10, type=int, help='Number of transformer head.')
    parser.add_argument('--head_dim', default=20, type=int, help='Head dimension of transformer.')
    parser.add_argument('--tf_drop_rate', default=0.1, type=float, help='Transformer dropout rate.')
    parser.add_argument('--emb_drop_rate', default=0.5, type=float, help='Embedding dropout rate.')
    parser.add_argument('--cell_drop_rate', default=0.2, type=float, help='Aggregation module dropout rate.')
    parser.add_argument('--word_alphabet_size', type=int, help='Word alphabet size.')
    parser.add_argument('--char_alphabet_size', type=int, help='Char alphabet size.')
    parser.add_argument('--label_alphabet_size', type=int, help='Label alphabet size.')
    parser.add_argument('--char_dim', type=int, help='Char embedding size.')
    parser.add_argument('--word_dim', type=int, help='Word embedding size.')
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)

    args = parser.parse_args()

    status = args.status.lower()
    seed_num = args.seed
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    output_file = args.output
    saved_set_path = args.saved_set
    saved_model_path = args.saved_model
    char_file = args.char_emb
    word_file = args.word_emb

    if status == 'train':
        if os.path.exists(saved_set_path):
            print('Loading saved data set...')
            with open(saved_set_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = Data()
            data_initialization(data, word_file, train_file, dev_file, test_file)
            data.generate_instance_with_words(train_file, 'train')
            data.generate_instance_with_words(dev_file, 'dev')
            data.generate_instance_with_words(test_file, 'test')
            data.build_char_pretrain_emb(char_file)
            data.build_word_pretrain_emb(word_file)
            if saved_set_path is not None:
                print('Dumping data...')
                with open(saved_set_path, 'wb') as f:
                    pickle.dump(data, f)
        data.show_data_summary()
        args.word_alphabet_size = data.word_alphabet.size()
        args.char_alphabet_size = data.char_alphabet.size()
        args.label_alphabet_size = data.label_alphabet.size()
        args.char_dim = data.char_emb_dim
        args.word_dim = data.word_emb_dim
        print_args(args)
        train(data, args, saved_model_path)

    elif status == 'test':
        assert not (test_file is None)
        if os.path.exists(saved_set_path):
            print('Loading saved data set...')
            with open(saved_set_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print("Cannot find saved data set: ", saved_set_path)
            exit(0)
        data.generate_instance_with_words(test_file, 'test')
        with open(saved_model_path + "_best_HP.config", "rb") as f:
            args = pickle.load(f)
        data.show_data_summary()
        print_args(args)
        load_model_decode(saved_model_path, data, args, "test")

    elif status == 'decode':
        assert not (raw_file is None or output_file is None)
        if os.path.exists(saved_set_path):
            print('Loading saved data set...')
            with open(saved_set_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print("Cannot find saved data set: ", saved_set_path)
            exit(0)
        data.generate_instance_with_words(raw_file, 'raw')
        with open(saved_model_path + "_best_HP.config", "rb") as f:
            args = pickle.load(f)
        data.show_data_summary()
        print_args(args)
        decode_results = load_model_decode(saved_model_path, data, args, "raw")
        data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")
