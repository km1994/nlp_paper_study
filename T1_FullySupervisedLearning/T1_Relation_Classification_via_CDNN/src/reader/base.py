import os
import re
import numpy as np
import tensorflow as tf
from collections import defaultdict
from collections import namedtuple


PAD_WORD = "<pad>"

Raw_Example = namedtuple('Raw_Example', 'label entity1 entity2 sentence')
PositionPair = namedtuple('PosPair', 'first last')

FLAGS = tf.app.flags.FLAGS # load FLAGS.word_dim

def load_raw_data(filename):
  '''load raw data from text file, 

  return: a list of Raw_Example
  '''
  data = []
  with open(filename) as f:
    for line in f:
      words = line.strip().split(' ')
      
      sent = words[5:]
      n = len(sent)
      if FLAGS.max_len < n:
        FLAGS.max_len = n

      label = int(words[0])

      entity1 = PositionPair(int(words[1]), int(words[2]))
      entity2 = PositionPair(int(words[3]), int(words[4]))

      example = Raw_Example(label, entity1, entity2, sent)
      data.append(example)
  print(FLAGS.max_len)
  return data

def maybe_build_vocab(raw_train_data, raw_test_data, vocab_file):
  '''collect words in sentence'''
  if not os.path.exists(vocab_file):
    vocab = set()
    for example in raw_train_data + raw_test_data:
      for w in example.sentence:
          vocab.add(w)

    with open(vocab_file, 'w') as f:
      for w in sorted(list(vocab)):
        f.write('%s\n' % w)
      f.write('%s\n' % PAD_WORD)

def _load_vocab(vocab_file):
  # load vocab from file
  vocab = []
  with open(vocab_file) as f:
    for line in f:
      w = line.strip()
      vocab.append(w)

  return vocab

def _load_embedding(embed_file, words_file):
  embed = np.load(embed_file)

  words2id = {}
  words = _load_vocab(words_file)
  for id, w in enumerate(words):
    words2id[w] = id
  
  return embed, words2id

def maybe_trim_embeddings(vocab_file, 
                        pretrain_embed_file,
                        pretrain_words_file,
                        trimed_embed_file):
  '''trim unnecessary words from original pre-trained word embedding

  Args:
    vocab_file: a file of tokens in train and test data
    pretrain_embed_file: file name of the original pre-trained embedding
    pretrain_words_file: file name of the words list w.r.t the embed
    trimed_embed_file: file name of the trimmed embedding
  '''
  if not os.path.exists(trimed_embed_file):
    pretrain_embed, pretrain_words2id = _load_embedding(
                                              pretrain_embed_file,
                                              pretrain_words_file)
    word_embed=[]
    vocab = _load_vocab(vocab_file)
    for w in vocab:
      if w in pretrain_words2id:
        id = pretrain_words2id[w]
        word_embed.append(pretrain_embed[id])
      else:
        vec = np.random.normal(0,0.1,[FLAGS.word_dim])
        word_embed.append(vec)
    pad_id = -1
    word_embed[pad_id] = np.zeros([FLAGS.word_dim])

    word_embed = np.asarray(word_embed)
    np.save(trimed_embed_file, word_embed.astype(np.float32))
    
  
  word_embed, vocab2id = _load_embedding(trimed_embed_file, vocab_file)
  return word_embed, vocab2id

def map_words_to_id(raw_data, word2id):
  '''inplace convert sentence from a list of words to a list of ids
  Args:
    raw_data: a list of Raw_Example
    word2id: dict, {word: id, ...}
  '''
  pad_id = word2id[PAD_WORD]
  for raw_example in raw_data:
    for idx, word in enumerate(raw_example.sentence):
      raw_example.sentence[idx] = word2id[word]

    # pad the sentence to FLAGS.max_len
    pad_n = FLAGS.max_len - len(raw_example.sentence)
    raw_example.sentence.extend(pad_n*[pad_id])

def _lexical_feature(raw_example):
  def _entity_context(e_idx, sent):
    ''' return [w(e-1), w(e), w(e+1)]
    '''
    context = []
    context.append(sent[e_idx])

    if e_idx >= 1:
      context.append(sent[e_idx-1])
    else:
      context.append(sent[e_idx])
    
    if e_idx < len(sent)-1:
      context.append(sent[e_idx+1])
    else:
      context.append(sent[e_idx])
    
    return context

    
  e1_idx = raw_example.entity1.first
  e2_idx = raw_example.entity2.first

  context1 = _entity_context(e1_idx, raw_example.sentence)
  context2 = _entity_context(e2_idx, raw_example.sentence)

  # ignore WordNet hypernyms in paper
  lexical = context1 + context2
  return lexical

def _position_feature(raw_example):
  def distance(n):
    '''convert relative distance to positive number
    -60), [-60, 60], (60
    '''
    # FIXME: FLAGS.pos_num
    if n < -60:
      return 0
    elif n >= -60 and n <= 60:
      return n + 61
    
    return 122

  e1_idx = raw_example.entity1.first
  e2_idx = raw_example.entity2.first

  position1 = []
  position2 = []
  length = len(raw_example.sentence)
  for i in range(length):
    position1.append(distance(i-e1_idx))
    position2.append(distance(i-e2_idx))
  
  return position1, position2

def build_sequence_example(raw_example):
  '''build tf.train.SequenceExample from Raw_Example
  context features : lexical, rid, direction (mtl)
  sequence features: sentence, position1, position2

  Args: 
    raw_example : type Raw_Example

  Returns:
    tf.trian.SequenceExample
  '''
  ex = tf.train.SequenceExample()

  lexical = _lexical_feature(raw_example)
  ex.context.feature['lexical'].int64_list.value.extend(lexical)

  rid = raw_example.label
  ex.context.feature['rid'].int64_list.value.append(rid)

  for word_id in raw_example.sentence:
    word = ex.feature_lists.feature_list['sentence'].feature.add()
    word.int64_list.value.append(word_id)
  
  position1, position2 = _position_feature(raw_example)
  for pos_val in position1:
    pos = ex.feature_lists.feature_list['position1'].feature.add()
    pos.int64_list.value.append(pos_val)
  for pos_val in position2:
    pos = ex.feature_lists.feature_list['position2'].feature.add()
    pos.int64_list.value.append(pos_val)

  return ex

def maybe_write_tfrecord(raw_data, filename):
  '''if the destination file is not exist on disk, convert the raw_data to 
  tf.trian.SequenceExample and write to file.

  Args:
    raw_data: a list of 'Raw_Example'
  '''
  if not os.path.exists(filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for raw_example in raw_data:
      example = build_sequence_example(raw_example)
      writer.write(example.SerializeToString())
    writer.close()

def _parse_tfexample(serialized_example):
  '''parse serialized tf.train.SequenceExample to tensors
  context features : lexical, rid, direction (mtl)
  sequence features: sentence, position1, position2
  '''
  context_features={
                      'lexical'   : tf.FixedLenFeature([6], tf.int64),
                      'rid'    : tf.FixedLenFeature([], tf.int64)}
  sequence_features={
                      'sentence' : tf.FixedLenSequenceFeature([], tf.int64),
                      'position1'  : tf.FixedLenSequenceFeature([], tf.int64),
                      'position2'  : tf.FixedLenSequenceFeature([], tf.int64)}
  context_dict, sequence_dict = tf.parse_single_sequence_example(
                      serialized_example,
                      context_features   = context_features,
                      sequence_features  = sequence_features)

  sentence = sequence_dict['sentence']
  position1 = sequence_dict['position1']
  position2 = sequence_dict['position2']

  lexical = context_dict['lexical']
  rid = context_dict['rid']

  return lexical, rid, sentence, position1, position2

def read_tfrecord_to_batch(filename, epoch, batch_size, pad_value, shuffle=True):
  '''read TFRecord file to get batch tensors for tensorflow models

  Returns:
    a tuple of batched tensors
  '''
  with tf.device('/cpu:0'):
    dataset = tf.data.TFRecordDataset([filename])
    # Parse the record into tensors
    dataset = dataset.map(_parse_tfexample) 
    dataset = dataset.repeat(epoch)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=100)
    
    # [] for no padding, [None] for padding to maximum length
    # n = FLAGS.max_len
    # if FLAGS.model == 'mtl':
    #   # lexical, rid, direction, sentence, position1, position2
    #   padded_shapes = ([None,], [], [], [n], [n], [n])
    # else:
    #   # lexical, rid, sentence, position1, position2
    #   padded_shapes = ([None,], [], [n], [n], [n])
    # pad_value = tf.convert_to_tensor(pad_value)
    # dataset = dataset.padded_batch(batch_size, padded_shapes,
    #                                padding_values=pad_value)
    dataset = dataset.batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    return batch


def inputs():
  raw_train_data = load_raw_data(FLAGS.train_file)
  raw_test_data = load_raw_data(FLAGS.test_file)

  maybe_build_vocab(raw_train_data, raw_test_data, FLAGS.vocab_file)

  if FLAGS.word_dim == 50:
    word_embed, vocab2id = maybe_trim_embeddings(
                                        FLAGS.vocab_file,
                                        FLAGS.senna_embed50_file,
                                        FLAGS.senna_words_file,
                                        FLAGS.trimmed_embed50_file)
  elif FLAGS.word_dim == 300:
    word_embed, vocab2id = maybe_trim_embeddings(
                                        FLAGS.vocab_file,
                                        FLAGS.google_embed300_file,
                                        FLAGS.google_words_file,
                                        FLAGS.trimmed_embed300_file)

  # map words to ids
  map_words_to_id(raw_train_data, vocab2id)
  map_words_to_id(raw_test_data, vocab2id)

  # convert raw data to TFRecord format data, and write to file
  train_record = FLAGS.train_record
  test_record = FLAGS.test_record
  
  maybe_write_tfrecord(raw_train_data, train_record)
  maybe_write_tfrecord(raw_test_data, test_record)

  pad_value = vocab2id[PAD_WORD]
  train_data = read_tfrecord_to_batch(train_record, 
                              FLAGS.num_epochs, FLAGS.batch_size, 
                              pad_value, shuffle=True)
  test_data = read_tfrecord_to_batch(test_record, 
                              FLAGS.num_epochs, 2717, 
                              pad_value, shuffle=False)

  return train_data, test_data, word_embed

def write_results(predictions, relations_file, results_file):
  relations = []
  with open(relations_file) as f:
    for line in f:
      segment = line.strip().split()
      relations.append(segment[1])
  
  start_no = 8001
  with open(results_file, 'w') as f:
    for idx, id in enumerate(predictions):
      rel = relations[id]
      f.write('%d\t%s\n' % (start_no+idx, rel))
