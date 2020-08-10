#! /usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from input_helpers import InputHelper
from siamese_network_semantic import SiameseLSTMw2v
import gzip
from random import random
import sys

# a=['你好','上帝','下地']
# b=[u'你好',u'上帝',u'下地']
# print(a)
# print(b)
# sys.exit(0)

# Parameters
# word2vec模型（采用已训练好的中文模型）
WORD2VEC_MODEL = '../word2vecmodel/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
# 　模型格式为bin
WORD2VEC_FORMAT = 'bin'
# word2vec词嵌入维数（64/128可选）
EMBEDDING_DIM = 64
# dropout比例设置
# DROPOUT_KEEP_PROB = '0.3'#训练集的拟合能力不够
DROPOUT_KEEP_PROB = '0.8'
# DROPOUT_KEEP_PROB = '0.6'
# DROPOUT_KEEP_PROB = '0.7'
# DROPOUT_KEEP_PROB = '0.8'
# DROPOUT_KEEP_PROB = '1.0'(7th-June)
# DROPOUT_KEEP_PROB = '0.8'
# DROPOUT_KEEP_PROB = '0.4'#训练集的拟合能力不够
# L2正规化系数(目前暂未生效)
L2_REG_LAMBDA = 0.0
# 原始训练文件
TRAINING_FILES_RAW = './train_data/atec_nlp_sim_train.csv'
# 隐藏层单元数
# HIDDEN_UNITS = 64(7th-June)
HIDDEN_UNITS = 128

# Training parameters
# 批大小
# BATCH_SIZE = 64
# BATCH_SIZE = 1024(7th-June)
BATCH_SIZE = 1024  # 92229=102477-10248
# epoch数目
# NUM_EPOCHS = 300
# NUM_EPOCHS = 3000
NUM_EPOCHS = 100000
# 模型评估周期（每隔多少步）
# EVALUATE_EVERY = 10(7th-June)
EVALUATE_EVERY = 100
# EVALUATE_EVERY = 10
# 模型保存周期(每隔多少步)
# CHECKOUTPOINT_EVERY = 1000
# CHECKOUTPOINT_EVERY = 10000
# CHECKOUTPOINT_EVERY = 1000(7th-Jnue)
CHECKOUTPOINT_EVERY = 1000
# 语句最多长度(包含多少个词)
# MAX_DOCUMENT_LENGTH = 12
# MAX_DOCUMENT_LENGTH = 8
# MAX_DOCUMENT_LENGTH = 20(7th-June)
MAX_DOCUMENT_LENGTH = 40
# 验证集比例
DEV_PERCENT = 10

# Misc Parameters
ALLOW_SOFT_PLACEMENT = True
LOG_DEVICE_PLACEMENT = False

print ('训练开始......................')
start_time = datetime.datetime.now()

inpH = InputHelper()
# 将原始的训练文件转化为分词后的训练文件
# inpH.train_file_preprocess(TRAINING_FILES_RAW, TRAINING_FILES_FORMAT)
# sys.exit(0)


train_set, dev_set, vocab_processor, sum_no_of_batches = inpH.getDataSets(TRAINING_FILES_RAW, MAX_DOCUMENT_LENGTH,
                                                                          DEV_PERCENT,
                                                                          BATCH_SIZE)

# dev_batches = inpH.batch_iter(list(zip(dev_set[0], dev_set[1], dev_set[2])), BATCH_SIZE, 1)
# for index,dev_batch in enumerate(dev_batches):
#     print(index, dev_batch)
# sys.exit(0)

# for index, value in enumerate(dev_set[2]):
#     print(index, dev_set[0][index], dev_set[1][index], dev_set[2][index])
# sys.exit(0)

# for index, w in enumerate(vocab_processor.vocabulary_._mapping):
#     print('vocab-{}:{}'.format(index, w))
# sys.exit(0)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=ALLOW_SOFT_PLACEMENT,
        log_device_placement=LOG_DEVICE_PLACEMENT)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        siameseModel = SiameseLSTMw2v(
            sequence_length=MAX_DOCUMENT_LENGTH,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=EMBEDDING_DIM,
            hidden_units=HIDDEN_UNITS,
            l2_reg_lambda=L2_REG_LAMBDA,
            batch_size=BATCH_SIZE,
            trainableEmbeddings=False
        )
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)

    grads_and_vars = optimizer.compute_gradients(siameseModel.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", siameseModel.loss)
    acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)
    f1_summary = tf.summary.scalar('f1', siameseModel.f1)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    # Write vocabulary
    vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)

    # 加载word2vec
    inpH.loadW2V(WORD2VEC_MODEL, WORD2VEC_FORMAT)
    # initial matrix with random uniform
    # initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), EMBEDDING_DIM))
    initW = np.random.uniform(0, 0, (len(vocab_processor.vocabulary_), EMBEDDING_DIM))
    # print(initW)
    # sys.exit(0)

    # load any vectors from the word2vec
    print("initializing initW with pre-trained word2vec embeddings")
    for index, w in enumerate(vocab_processor.vocabulary_._mapping):
        # print('vocab-{}:{}'.format(index, w))

        arr = []
        if w in inpH.pre_emb:
            arr = inpH.pre_emb[w]
            # print('=====arr-{},{}'.format(index, arr))
            idx = vocab_processor.vocabulary_.get(w)
            initW[idx] = np.asarray(arr).astype(np.float32)

        # 不使用词向量
        # arr=[]
        # idx = vocab_processor.vocabulary_.get(w)
        # arr.append(idx)
        # initW[idx] = np.asarray(arr).astype(np.float32)

    print("Done assigning intiW. len=" + str(len(initW)))
    # exit(0)

    # for idx, value in enumerate(initW):
    #     print(idx, value)
    # sys.exit(0)

    inpH.deletePreEmb()
    gc.collect()
    sess.run(siameseModel.W.assign(initW))


    def train_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        # for index, sentence in enumerate(x1_batch):
        #     word_list1=[]
        #     word_list2=[]
        #     y=y_batch[index]
        #     for idx in x1_batch[index]:
        #         word_list1.append(vocab_processor.vocabulary_.reverse(idx))
        #     for idx in x2_batch[index]:
        #         word_list2.append(vocab_processor.vocabulary_.reverse(idx))
        #
        #     # print(''.join(word_list1),'\t',''.join(word_list2),'\t',y)
        #     print('==========={}=============='.format(index))
        #     print(''.join(word_list1))
        #     print (''.join(word_list2))
        #     print(y)
        # sys.exit(0)

        feed_dict = {
            siameseModel.input_x1: x1_batch,
            siameseModel.input_x2: x2_batch,
            siameseModel.input_y: y_batch,
            siameseModel.dropout_keep_prob: DROPOUT_KEEP_PROB,
        }
        _, step, loss, accuracy, f1, dist, sim, summaries = sess.run(
            [tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.f1, siameseModel.distance,
             siameseModel.temp_sim, train_summary_op], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("TRAIN {}: step {}, loss {:g}, acc {:g}, f1 {:g}".format(time_str, step, loss, accuracy, f1))
        train_summary_writer.add_summary(summaries, step)
        print(y_batch, dist, sim)


    def dev_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        # for index, sentence in enumerate(x1_batch):
        #     word_list1=[]
        #     word_list2=[]
        #     y=y_batch[index]
        #     for idx in x1_batch[index]:
        #         word_list1.append(vocab_processor.vocabulary_.reverse(idx))
        #     for idx in x2_batch[index]:
        #         word_list2.append(vocab_processor.vocabulary_.reverse(idx))
        #
        #     # print(''.join(word_list1),'\t',''.join(word_list2),'\t',y)
        #     print('==========={}=============='.format(index))
        #     print(''.join(word_list1))
        #     print (''.join(word_list2))
        #     print(y)
        # sys.exit(0)

        feed_dict = {
            siameseModel.input_x1: x2_batch,
            siameseModel.input_x2: x1_batch,
            siameseModel.input_y: y_batch,
            siameseModel.dropout_keep_prob: 1.0,
        }
        step, loss, accuracy, f1, sim, summaries = sess.run(
            [global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.f1, siameseModel.temp_sim,
             dev_summary_op], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("DEV {}: step {}, loss {:g}, acc {:g}, f1 {:g}".format(time_str, step, loss, accuracy, f1))
        dev_summary_writer.add_summary(summaries, step)
        print (y_batch, sim)
        return accuracy


    ##################
    # sys.exit(0)

    # Generate batches
    batches = inpH.batch_iter(
        list(zip(train_set[0], train_set[1], train_set[2])), BATCH_SIZE, NUM_EPOCHS)

    ptr = 0
    max_validation_acc = 0.0
    for nn in xrange(sum_no_of_batches * NUM_EPOCHS):
        batch = batches.next()
        if len(batch) < 1:
            continue
        x1_batch, x2_batch, y_batch = zip(*batch)
        if len(y_batch) < 1:
            continue
        train_step(x1_batch, x2_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc = 0.0
        cnt = 0
        if current_step % EVALUATE_EVERY == 0:
            print("\nEvaluation:")
            dev_batches = inpH.batch_iter(list(zip(dev_set[0], dev_set[1], dev_set[2])), BATCH_SIZE, 1)
            for db in dev_batches:
                if len(db) < 1:
                    continue
                x1_dev_b, x2_dev_b, y_dev_b = zip(*db)
                if len(y_dev_b) < 1:
                    continue
                acc = dev_step(x1_dev_b, x2_dev_b, y_dev_b)
                sum_acc = sum_acc + acc
                cnt += 1

            sum_acc /= cnt
            print("sum_acc= {}".format(sum_acc))
        if current_step % CHECKOUTPOINT_EVERY == 0:
            if sum_acc >= max_validation_acc:
                max_validation_acc = sum_acc

            # 临时逻辑
            saver.save(sess, checkpoint_prefix, global_step=current_step)
            tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph" + str(nn) + ".pb",
                                 as_text=False)
            print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc,
                                                                                  checkpoint_prefix))

        print('max_validation_acc(each batch)= {}'.format(max_validation_acc))

end_time = datetime.datetime.now()
train_duration = end_time - start_time
print('训练开始时间: {}'.format(start_time))
print('训练结束时间: {}'.format(end_time))
print('训练结束, 训练总耗时: {}'.format(train_duration))
