#coding=utf8
from data_helper import load_set, batch_iter
import embedding as emb
from model import *
import time
import os
import datetime
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging

# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
timestamp = str(int(time.time()))
fh = logging.FileHandler('./log/log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

tf.app.flags.DEFINE_integer('embedding_dim', 100, 'The dimension of the word embedding')
tf.app.flags.DEFINE_integer('num_filters_A', 50, 'The number of filters in block A')
tf.app.flags.DEFINE_integer('num_filters_B', 50, 'The number of filters in block B')
tf.app.flags.DEFINE_integer('n_hidden', 150, 'number of hidden units in the fully connected layer')
tf.app.flags.DEFINE_integer('sentence_length', 100, 'max size of sentence')
tf.app.flags.DEFINE_integer('num_classes', 6, 'num of the labels')
tf.flags.DEFINE_float("l2_reg_lambda", 1, "L2 regularization lambda (default: 0.0)")

tf.app.flags.DEFINE_integer('num_epochs', 85, 'Number of epochs to be trained')
tf.app.flags.DEFINE_integer('batch_size', 64, 'size of mini batch')

tf.app.flags.DEFINE_integer("display_step", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

tf.app.flags.DEFINE_float('lr', 1e-3, 'learning rate')

tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

filter_size = [1, 2, 100]
conf = tf.app.flags.FLAGS
# conf._parse_flags()

#glove是载入的次向量。glove.d是单词索引字典<word, index>，glove.g是词向量矩阵<词个数,300>
print('loading glove...')
glove = emb.GloVe(N=100)

# print("Loading data...")
Xtrain, ytrain = load_set(glove, path='./sts/semeval-sts/all')
Xtrain[0], Xtrain[1], ytrain = shuffle(Xtrain[0], Xtrain[1], ytrain)
#[22592, 句长]
Xtest, ytest = load_set(glove, path='./sts/semeval-sts/2016')
Xtest[0], Xtest[1], ytest = shuffle(Xtest[0], Xtest[1], ytest)
#[1186, 句长]



# max_sent_length = max([len(x) for SS in Xtrain for x in SS])
# print max_sent_length #最大的句子长度为84
#-------------------------------------Loading finished----------------------------------------------#

#-------------------------------------training the network----------------------------------------------#
with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    input_1 = tf.placeholder(tf.int32, [None, conf.sentence_length], name="input_x1")
    input_2 = tf.placeholder(tf.int32, [None, conf.sentence_length], name="input_x2")
    input_3 = tf.placeholder(tf.float32, [None, conf.num_classes], name="input_y")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    with tf.name_scope("embendding"):
        s0_embed = tf.nn.embedding_lookup(glove.g, input_1)
        s1_embed = tf.nn.embedding_lookup(glove.g, input_2)

    with tf.name_scope("reshape"):
        input_x1 = tf.reshape(s0_embed, [-1, conf.sentence_length, conf.embedding_dim, 1])
        input_x2 = tf.reshape(s1_embed, [-1, conf.sentence_length, conf.embedding_dim, 1])
        input_y = tf.reshape(input_3, [-1, conf.num_classes])

    # sent1_unstack = tf.unstack(input_x1, axis=1)
    # sent2_unstack = tf.unstack(input_x2, axis=1)
    # D = []
    # for i in range(len(sent1_unstack)):
    #     d = []
    #     for j in range(len(sent2_unstack)):
    #         dis = compute_cosine_distance(sent1_unstack[i], sent2_unstack[j])
    #         d.append(dis)
    #     D.append(d)
    # D = tf.reshape(D, [-1, len(sent1_unstack), len(sent2_unstack), 1])
    # A = [tf.nn.softmax(tf.expand_dims(tf.reduce_sum(D, axis=i), 2)) for i in [2, 1]]
    #
    # print A[1]
    # print A[1] * input_x2
    # atten_embed = tf.concat([input_x2, A[1] * input_x2], 2)

    setence_model = MPCNN_Layer(conf.num_classes, conf.embedding_dim, filter_size,
                                [conf.num_filters_A, conf.num_filters_B], conf.n_hidden,
                                input_x1, input_x2, input_y, dropout_keep_prob, conf.l2_reg_lambda)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    setence_model.similarity_measure_layer()
    optimizer = tf.train.AdamOptimizer(conf.lr)
    grads_and_vars = optimizer.compute_gradients(setence_model.loss)
    train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    # print("Writing to {}\n".format(out_dir))
    #
    loss_summary = tf.summary.scalar("loss", setence_model.loss)
    acc_summary = tf.summary.scalar("accuracy", setence_model.accuracy)
    #
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    #
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
    #
    # checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=conf.num_checkpoints)

    def train(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
          input_1: x1_batch,
          input_2: x2_batch,
          input_3: y_batch,
          dropout_keep_prob: 0.5
        }
        _, step, summaries, batch_loss, accuracy = sess.run(
            [train_step, global_step, train_summary_op, setence_model.loss, setence_model.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, batch_loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def dev_step(x1_batch, x2_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          input_1: x1_batch,
          input_2: x2_batch,
          input_3: y_batch,
          dropout_keep_prob: 1
        }
        _, step, summaries, batch_loss, accuracy = sess.run(
            [train_step, global_step, dev_summary_op, setence_model.loss, setence_model.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        dev_summary_writer.add_summary(summaries, step)
        # if writer:
        #     writer.add_summary(summaries, step)

        return batch_loss, accuracy

    sess.run(tf.global_variables_initializer())
    batches = batch_iter(list(zip(Xtrain[0], Xtrain[1], ytrain)), conf.batch_size, conf.num_epochs)
    for batch in batches:
        x1_batch, x2_batch, y_batch = zip(*batch)
        train(x1_batch, x2_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % conf.evaluate_every == 0:
            total_dev_loss = 0.0
            total_dev_accuracy = 0.0

            logger.info("\nEvaluation:")
            dev_batches = batch_iter(list(zip(Xtest[0], Xtest[1], ytest)), conf.batch_size, 1)
            for dev_batch in dev_batches:
                x1_dev_batch, x2_dev_batch, y_dev_batch = zip(*dev_batch)
                dev_loss, dev_accuracy = dev_step(x1_dev_batch, x2_dev_batch, y_dev_batch)
                total_dev_loss += dev_loss
                total_dev_accuracy += dev_accuracy
            total_dev_accuracy = total_dev_accuracy / (len(ytest) / conf.batch_size)
            logger.info("dev_loss {:g}, dev_acc {:g}, num_dev_batches {:g}".format(total_dev_loss, total_dev_accuracy,
                                                                             len(ytest) / conf.batch_size))
            # train_summary_writer.add_summary(summaries)

    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # for i in range(conf.num_epochs):
    #     training_batch = zip(range(0, len(Xtrain[0]), conf.batch_size),
    #                          range(conf.batch_size, len(Xtrain[0]) + 1, conf.batch_size))
    #     for start, end in training_batch:
    #         feed_dict = {input_1: Xtrain[0][start:end], input_2: Xtrain[1][start:end],
    #                      dropout_keep_prob: 0.5, input_3: ytrain[start:end]}
    #         print start
    #         #assert all(x.shape == (100, 100) for x in Xtrain[0][start:end])
    #         loss, _ = sess.run(train_step, feed_dict=feed_dict)
    #         print("Epoch:", '%04d' % (i + 1), "cost=", "{:.9f}".format(loss))

    logger.info("Optimization Finished!")