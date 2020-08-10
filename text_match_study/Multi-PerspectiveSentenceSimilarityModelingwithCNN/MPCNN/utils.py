import tensorflow as tf
import numpy as np

def compute_l1_distance(x, y):
    with tf.name_scope('l1_distance'):
        d = tf.reduce_sum(tf.abs(tf.subtract(x, y)), axis=1)
        return d

def compute_euclidean_distance(x, y):
    with tf.name_scope('euclidean_distance'):
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1))
        return d

def compute_pearson_distance(x, y):
    with tf.name_scope("pearson"):
        mid1 = tf.reduce_mean(x * y, axis=1) - \
                    tf.reduce_mean(x, axis=1) * tf.reduce_mean(y, axis=1)
        mid2 = tf.sqrt(tf.reduce_mean(tf.square(x), axis=1) - tf.square(tf.reduce_mean(x, axis=1))) * \
               tf.sqrt(tf.reduce_mean(tf.square(y), axis=1) - tf.square(tf.reduce_mean(y, axis=1)))
        return mid1 / mid2

def compute_cosine_distance(x, y):
    with tf.name_scope('cosine_distance'):
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
        x_y = tf.reduce_sum(tf.multiply(x, y), axis=1)
        d = tf.divide(x_y, tf.multiply(x_norm, y_norm))
        return d

def comU1(x, y):
    result = [compute_cosine_distance(x, y), compute_l1_distance(x, y)]
    # result = [compute_euclidean_distance(x, y), compute_euclidean_distance(x, y), compute_euclidean_distance(x, y)]
    return tf.stack(result, axis=1)


def comU2(x, y):
    # result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y)]
    # return tf.stack(result, axis=1)
    return tf.expand_dims(compute_cosine_distance(x, y), -1)
