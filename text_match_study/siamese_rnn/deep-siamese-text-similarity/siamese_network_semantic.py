# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class SiameseLSTMw2v(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
    """

    def stackedRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
        n_hidden = hidden_units
        n_layers = 3
        # n_layers = 6
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        # print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell

        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

            outputs, _ = tf.nn.static_rnn(lstm_fw_cell_m, x, dtype=tf.float32)
        return outputs[-1]

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2+reg

    def __init__(
            self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size,
            trainableEmbeddings):
        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.constant(0.0, shape=[vocab_size, embedding_size]),
                trainable=trainableEmbeddings, name="W")
            self.embedded_words1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_words2 = tf.nn.embedding_lookup(self.W, self.input_x2)
        # print self.embedded_words1
        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1 = self.stackedRNN(self.embedded_words1, self.dropout_keep_prob, "side1", embedding_size,
                                        sequence_length, hidden_units)
            self.out2 = self.stackedRNN(self.embedded_words2, self.dropout_keep_prob, "side2", embedding_size,
                                        sequence_length, hidden_units)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keep_dims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
                                        name="temp_sim")  # auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope('f1'):
            ones_like_actuals = tf.ones_like(self.input_y)
            zeros_like_actuals = tf.zeros_like(self.input_y)
            ones_like_predictions = tf.ones_like(self.temp_sim)
            zeros_like_predictions = tf.zeros_like(self.temp_sim)

            tp = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(self.input_y, ones_like_actuals),
                        tf.equal(self.temp_sim, ones_like_predictions)
                    ),
                    'float'
                )
            )

            tn = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(self.input_y, zeros_like_actuals),
                        tf.equal(self.temp_sim, zeros_like_predictions)
                    ),
                    'float'
                )
            )

            fp = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(self.input_y, zeros_like_actuals),
                        tf.equal(self.temp_sim, ones_like_predictions)
                    ),
                    'float'
                )
            )

            fn = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(self.input_y, ones_like_actuals),
                        tf.equal(self.temp_sim, zeros_like_predictions)
                    ),
                    'float'
                )
            )

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            self.f1 = 2 * precision * recall / (precision + recall)
