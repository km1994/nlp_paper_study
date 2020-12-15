from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import codecs
import collections
import shutil
import sys

import numpy as np
import tensorflow as tf
import pyhocon

def initialize_from_env():
  name = sys.argv[1]
  print("Running experiment: {}".format(name))

  config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
  config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))

  print(pyhocon.HOCONConverter.convert(config, "hocon"))
  return config

def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)

def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])

def flatten(l):
  return [item for sublist in l for item in sublist]


def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path

def load_char_dict(char_vocab_path):
  vocab = [u"<unk>"]
  with codecs.open(char_vocab_path, encoding="utf-8") as f:
    vocab.extend(l.strip() for l in f.readlines())
  char_dict = collections.defaultdict(int)
  char_dict.update({c:i for i, c in enumerate(vocab)})
  return char_dict

def maybe_divide(x, y):
  return 0 if y == 0 else x / float(y)

def projection(inputs, output_size, initializer=None):
  return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)

def highway(inputs, num_layers, dropout):
  for i in range(num_layers):
    with tf.variable_scope("highway_{}".format(i)):
      j, f = tf.split(projection(inputs, 2 * shape(inputs, -1)), 2, -1)
      f = tf.sigmoid(f)
      j = tf.nn.relu(j)
      if dropout is not None:
        j = tf.nn.dropout(j, dropout)
      inputs = f * j + (1 - f) * inputs
  return inputs

def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
  if len(inputs.get_shape()) > 3:
    raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

  if len(inputs.get_shape()) == 3:
    batch_size = shape(inputs, 0)
    seqlen = shape(inputs, 1)
    emb_size = shape(inputs, 2)
    current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
  else:
    current_inputs = inputs

  for i in range(num_hidden_layers):
    hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
    hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
    current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

    if dropout is not None:
      current_outputs = tf.nn.dropout(current_outputs, dropout)
    current_inputs = current_outputs

  output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
  output_bias = tf.get_variable("output_bias", [output_size])
  outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

  if len(inputs.get_shape()) == 3:
    outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
  return outputs

def cnn(inputs, filter_sizes, num_filters):
  input_size = shape(inputs, 2)
  outputs = []
  for i, filter_size in enumerate(filter_sizes):
    with tf.variable_scope("conv_{}".format(i)):
      w = tf.get_variable("w", [filter_size, input_size, num_filters])
      b = tf.get_variable("b", [num_filters])
    conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
    h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
    pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
    outputs.append(pooled)
  return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]

def bilinear_classifier(x_bnv, y_bnv, keep_prob, output_size = 1, add_bias_1=True, add_bias_2=True):
  """biaffine_mapping() with dropout."""


  # Statically known input dimensions.
  input_size = x_bnv.get_shape().as_list()[-1]

  # Dynamically known input dimensions
  batch_size = tf.shape(x_bnv)[0]
  noise_shape = [batch_size, 1, input_size]
  x_bnv = tf.nn.dropout(x_bnv, keep_prob, noise_shape=noise_shape)
  y_bnv = tf.nn.dropout(y_bnv, keep_prob, noise_shape=noise_shape)

  biaffine = biaffine_mapping(
      x_bnv,
      y_bnv,
      output_size,
      add_bias_1=add_bias_1,
      add_bias_2=add_bias_2,
      initializer=tf.zeros_initializer())
  if output_size == 1:
    output = tf.squeeze(biaffine,axis=2)
  else:
    output = tf.transpose(biaffine,[0,1,3,2])
  return output

def biaffine_mapping(vector_set_1,
               vector_set_2,
               output_size,
               add_bias_1=True,
               add_bias_2=True,
              initializer= None):
  """Bilinear mapping: maps two vector spaces to a third vector space.

  The input vector spaces are two 3d matrices: batch size x bucket size x values
  A typical application of the function is to compute a square matrix
  representing a dependency tree. The output is for each bucket a square
  matrix of the form [bucket size, output size, bucket size]. If the output size
  is set to 1 then results is [bucket size, 1, bucket size] equivalent to
  a square matrix where the bucket for instance represent the tokens on
  the x-axis and y-axis. In this way represent the adjacency matrix of a
  dependency graph (see https://arxiv.org/abs/1611.01734).

  Args:
     vector_set_1: vectors of space one
     vector_set_2: vectors of space two
     output_size: number of output labels (e.g. edge labels)
     add_bias_1: Whether to add a bias for input one
     add_bias_2: Whether to add a bias for input two
     initializer: Initializer for the bilinear weight map

  Returns:
    Output vector space as 4d matrix:
    batch size x bucket size x output size x bucket size
    The output could represent an unlabeled dependency tree when
    the output size is 1 or a labeled tree otherwise.

  """
  with tf.variable_scope('Bilinear'):
    # Dynamic shape info
    batch_size = tf.shape(vector_set_1)[0]
    bucket_size = tf.shape(vector_set_1)[1]

    if add_bias_1:
      vector_set_1 = tf.concat(
          [vector_set_1, tf.ones([batch_size, bucket_size, 1])], axis=2)
    if add_bias_2:
      vector_set_2 = tf.concat(
          [vector_set_2, tf.ones([batch_size, bucket_size, 1])], axis=2)

    # Static shape info
    vector_set_1_size = vector_set_1.get_shape().as_list()[-1]
    vector_set_2_size = vector_set_2.get_shape().as_list()[-1]

    if not initializer:
      initializer = tf.orthogonal_initializer()

    # Mapping matrix
    bilinear_map = tf.get_variable(
        'bilinear_map', [vector_set_1_size, output_size, vector_set_2_size],
        initializer=initializer)

    # The matrix operations and reshapings for bilinear mapping.
    # b: batch size (batch of buckets)
    # v1, v2: values (size of vectors)
    # n: tokens (size of bucket)
    # r: labels (output size), e.g. 1 if unlabeled or number of edge labels.

    # [b, n, v1] -> [b*n, v1]
    vector_set_1 = tf.reshape(vector_set_1, [-1, vector_set_1_size])

    # [v1, r, v2] -> [v1, r*v2]
    bilinear_map = tf.reshape(bilinear_map, [vector_set_1_size, -1])

    # [b*n, v1] x [v1, r*v2] -> [b*n, r*v2]
    bilinear_mapping = tf.matmul(vector_set_1, bilinear_map)

    # [b*n, r*v2] -> [b, n*r, v2]
    bilinear_mapping = tf.reshape(
        bilinear_mapping,
        [batch_size, bucket_size * output_size, vector_set_2_size])

    # [b, n*r, v2] x [b, n, v2]T -> [b, n*r, n]
    bilinear_mapping = tf.matmul(bilinear_mapping, vector_set_2, adjoint_b=True)

    # [b, n*r, n] -> [b, n, r, n]
    bilinear_mapping = tf.reshape(
        bilinear_mapping, [batch_size, bucket_size, output_size, bucket_size])
    return bilinear_mapping

class EmbeddingDictionary(object):
  def __init__(self, info, normalize=True, maybe_cache=None):
    self._size = info["size"]
    self._normalize = normalize
    self._path = info["path"]
    if maybe_cache is not None and maybe_cache._path == self._path:
      assert self._size == maybe_cache._size
      self._embeddings = maybe_cache._embeddings
    else:
      self._embeddings = self.load_embedding_dict(self._path)

  @property
  def size(self):
    return self._size

  def load_embedding_dict(self, path):
    print("Loading word embeddings from {}...".format(path))
    default_embedding = np.zeros(self.size)
    embedding_dict = collections.defaultdict(lambda:default_embedding)
    if len(path) > 0:
      vocab_size = None
      with open(path) as f:
        for i, line in enumerate(f.readlines()):
          word_end = line.find(" ")
          word = line[:word_end]
          embedding = np.fromstring(line[word_end + 1:], np.float32, sep=" ")
          assert len(embedding) == self.size
          embedding_dict[word] = embedding
      if vocab_size is not None:
        assert vocab_size == len(embedding_dict)
      print("Done loading word embeddings.")
    return embedding_dict

  def is_in_embeddings(self, key):
    return self._embeddings.has_key(key)

  def __getitem__(self, key):
    embedding = self._embeddings[key]
    if self._normalize:
      embedding = self.normalize(embedding)
    return embedding

  def normalize(self, v):
    norm = np.linalg.norm(v)
    if norm > 0:
      return v / norm
    else:
      return v

class CustomLSTMCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, batch_size, dropout):
    self._num_units = num_units
    self._dropout = dropout
    self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
    self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
    initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
    initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
    self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  @property
  def initial_state(self):
    return self._initial_state

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
      c, h = state
      h *= self._dropout_mask
      concat = projection(tf.concat([inputs, h], 1), 3 * self.output_size, initializer=self._initializer)
      i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
      i = tf.sigmoid(i)
      new_c = (1 - i) * c  + i * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)
      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      return new_h, new_state

  def _orthonormal_initializer(self, scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
      M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
      Q1, R1 = np.linalg.qr(M1)
      Q2, R2 = np.linalg.qr(M2)
      Q1 = Q1 * np.sign(np.diag(R1))
      Q2 = Q2 * np.sign(np.diag(R2))
      n_min = min(shape[0], shape[1])
      params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
      return params
    return _initializer

  def _block_orthonormal_initializer(self, output_sizes):
    def _initializer(shape, dtype=np.float32, partition_info=None):
      assert len(shape) == 2
      assert sum(output_sizes) == shape[1]
      initializer = self._orthonormal_initializer()
      params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
      return params
    return _initializer
