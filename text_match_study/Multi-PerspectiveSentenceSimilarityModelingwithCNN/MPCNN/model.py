import tensorflow as tf
from utils import *
import tensorflow.contrib.slim as slim

def init_weight(shape, name):
    var = tf.Variable(tf.truncated_normal(shape, mean=0, stddev=1.0), name=name)
    return var

class MPCNN_Layer():
    def __init__(self, num_classes, embedding_size, filter_sizes, num_filters, n_hidden,
                 input_x1, input_x2, input_y, dropout_keep_prob, l2_reg_lambda):
        '''

        :param sequence_length:
        :param num_classes:
        :param embedding_size:
        :param filter_sizes:
        :param num_filters:
        '''
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self .num_filters = num_filters
        self.num_classes = num_classes
        self.poolings = [tf.reduce_max, tf.reduce_min, tf.reduce_mean]

        self.input_x1 = input_x1
        self.input_x2 = input_x2
        self.input_y = input_y
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = l2_reg_lambda
        self.W1 = [init_weight([filter_sizes[0], embedding_size, 1, num_filters[0]], "W1_0"),
                   init_weight([filter_sizes[1], embedding_size, 1, num_filters[0]], "W1_1"),
                   init_weight([filter_sizes[2], embedding_size, 1, num_filters[0]], "W1_2")]
        self.b1 = [tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_0"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_1"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_2")]

        self.W2 = [init_weight([filter_sizes[0], embedding_size, 1, num_filters[1]], "W2_0"),
                   init_weight([filter_sizes[1], embedding_size, 1, num_filters[1]], "W2_1")]
        self.b2 = [tf.Variable(tf.constant(0.1, shape=[num_filters[1], embedding_size]), "b2_0"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[1], embedding_size]), "b2_1")]
        self.h = num_filters[0]*len(self.poolings)*2 + \
                 num_filters[1]*(len(self.poolings)-1)*(len(filter_sizes)-1)*3 + \
                 len(self.poolings)*len(filter_sizes)*len(filter_sizes)*3
        self.Wh = tf.Variable(tf.random_normal([604, n_hidden], stddev=0.01), name='Wh')
        self.bh = tf.Variable(tf.constant(0.1, shape=[n_hidden]), name="bh")

        self.Wo = tf.Variable(tf.random_normal([n_hidden, num_classes], stddev=0.01), name='Wo')
        self.bo = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bo")


    def attention(self):
        sent1_unstack = tf.unstack(self.input_x1, axis=1)
        sent2_unstack = tf.unstack(self.input_x2, axis=1)
        D = []
        for i in range(len(sent1_unstack)):
            d = []
            for j in range(len(sent2_unstack)):
                dis = compute_cosine_distance(sent1_unstack[i], sent2_unstack[j])
                #dis:[batch_size, 1(channels)]
                d.append(dis)
            D.append(d)
            print(i)
        D = tf.reshape(D, [-1, len(sent1_unstack), len(sent2_unstack), 1])
        A = [tf.nn.softmax(tf.expand_dims(tf.reduce_sum(D, axis=i), 2)) for i in [2, 1]]
        atten_embed = []
        atten_embed.append(tf.concat([self.input_x1, A[0] * self.input_x1], 2))
        atten_embed.append(tf.concat([self.input_x2, A[1] * self.input_x2], 2))
        return atten_embed

    def per_dim_conv_layer(self, x, w, b, pooling):
        '''

        :param input: [batch_size, sentence_length, embed_size, 1]
        :param w: [ws, embedding_size, 1, num_filters]
        :param b: [num_filters, embedding_size]
        :param pooling:
        :return:
        '''
        # unpcak the input in the dim of embed_dim
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        for i in range(x.get_shape()[2]):
            conv = tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID")
            conv = slim.batch_norm(inputs=conv, activation_fn=tf.nn.tanh, is_training=self.is_training)
            convs.append(conv)
        conv = tf.stack(convs, axis=2)
        pool = pooling(conv, axis=1)

        return pool

    def bulit_block_A(self, x):
        #bulid block A and cal the similarity according to algorithm 1
        out = []
        with tf.name_scope("bulid_block_A"):
            for pooling in self.poolings:
                pools = []
                for i, ws in enumerate(self.filter_sizes):
                    with tf.name_scope("conv-pool-%s" %ws):
                        conv = tf.nn.conv2d(x, self.W1[i], strides=[1, 1, 1, 1], padding="VALID")
                        conv = slim.batch_norm(inputs=conv, activation_fn=tf.nn.tanh, is_training=self.is_training)
                        pool = pooling(conv, axis=1)
                    pools.append(pool)
                out.append(pools)
            return out

    def bulid_block_B(self, x):
        out = []
        with tf.name_scope("bulid_block_B"):
            for pooling in self.poolings[:-1]:
                pools = []
                with tf.name_scope("conv-pool"):
                    for i, ws in enumerate(self.filter_sizes[:-1]):
                        with tf.name_scope("per_conv-pool-%s" % ws):
                            pool = self.per_dim_conv_layer(x, self.W2[i], self.b2[i], pooling)
                        pools.append(pool)
                    out.append(pools)
            return out


    def similarity_sentence_layer(self):
        # atten = self.attention() #[batch_size, length, 2*embedding, 1]
        sent1 = self.bulit_block_A(self.input_x1)
        sent2 = self.bulit_block_A(self.input_x2)
        fea_h = []
        with tf.name_scope("cal_dis_with_alg1"):
            for i in range(3):
                regM1 = tf.concat(sent1[i], 1)
                regM2 = tf.concat(sent2[i], 1)
                for k in range(self.num_filters[0]):
                    fea_h.append(comU2(regM1[:, :, k], regM2[:, :, k]))

        #self.fea_h = fea_h

        fea_a = []
        with tf.name_scope("cal_dis_with_alg2_2-9"):
            for i in range(3):
                for j in range(len(self.filter_sizes)):
                    for k in range(len(self.filter_sizes)):
                        fea_a.append(comU1(sent1[i][j][:, 0, :], sent2[i][k][:, 0, :]))
        #
        sent1 = self.bulid_block_B(self.input_x1)
        sent2 = self.bulid_block_B(self.input_x2)

        fea_b = []
        with tf.name_scope("cal_dis_with_alg2_last"):
            for i in range(len(self.poolings)-1):
                for j in range(len(self.filter_sizes)-1):
                    for k in range(self.num_filters[1]):
                        fea_b.append(comU1(sent1[i][j][:, :, k], sent2[i][j][:, :, k]))
        #self.fea_b = fea_b
        return tf.concat(fea_h + fea_a + fea_b, 1)


    def similarity_measure_layer(self, is_training=True):
        self.is_training = is_training
        fea = self.similarity_sentence_layer()
        self.h_drop = tf.nn.dropout(fea, self.dropout_keep_prob)
        # fea_h.extend(fea_a)
        # fea_h.extend(fea_b)
        #print len(fea_h), fea_h
        #fea = tf.concat(fea_h+fea_a+fea_b, 1)
        #print fea.get_shape()
        with tf.name_scope("full_connect_layer"):
            h = tf.nn.tanh(tf.matmul(fea, self.Wh) + self.bh)
            # h = tf.nn.dropout(h, self.dropout_keep_prob)
            self.scores = tf.matmul(h, self.Wo) + self.bo
            self.output = tf.nn.softmax(self.scores)
        #     return o

        # CalculateMean cross-entropy loss
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
        with tf.name_scope("loss"):
            # self.loss = -tf.reduce_sum(self.input_y * tf.log(self.output))
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.input_y, self.output))) + reg

            # self.loss = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))
            # self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.input_y, 1), tf.argmax(self.scores, 1)), tf.float32))


