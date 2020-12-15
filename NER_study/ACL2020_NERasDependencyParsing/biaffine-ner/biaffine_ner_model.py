from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,time,json,threading
import random
import numpy as np
import tensorflow as tf
import h5py

import util


class BiaffineNERModel():
  def __init__(self, config):
    self.config = config
    self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    self.context_embeddings_size = self.context_embeddings.size

    self.char_embedding_size = config["char_embedding_size"]
    self.char_dict = util.load_char_dict(config["char_vocab_path"])

    self.lm_file = h5py.File(self.config["lm_path"], "r")
    self.lm_layers = self.config["lm_layers"]
    self.lm_size = self.config["lm_size"]

    self.eval_data = None  # Load eval data lazily.
    self.ner_types = self.config['ner_types']
    self.ner_maps = {ner: (i + 1) for i, ner in enumerate(self.ner_types)}
    self.num_types = len(self.ner_types)

    input_props = []
    input_props.append((tf.string, [None, None]))  # Tokens.
    input_props.append((tf.float32, [None, None, self.context_embeddings_size]))  # Context embeddings.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers]))  # LM embeddings.
    input_props.append((tf.int32, [None, None, None]))  # Character indices.
    input_props.append((tf.int32, [None]))  # Text lengths.
    input_props.append((tf.bool, []))  # Is training.
    input_props.append((tf.int32, [None]))  # Gold NER Label

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.predictions, self.loss = self.get_predictions_and_loss(self.input_tensors)
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"],
                                               staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam": tf.train.AdamOptimizer,
      "sgd": tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

  def start_enqueue_thread(self, session):
    with open(self.config["train_path"]) as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

    def _enqueue_loop():
      while True:
        random.shuffle(train_examples)
        for example in train_examples:
          tensorized_example = self.tensorize_example(example, is_training=True)
          feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
          session.run(self.enqueue_op, feed_dict=feed_dict)
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def restore(self, session):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)

  def load_lm_embeddings(self, doc_key):
    if self.lm_file is None:
      return np.zeros([0, 0, self.lm_size, self.lm_layers])
    file_key = doc_key.replace("/", ":")
    if not file_key in self.lm_file and file_key[:-2] in self.lm_file:
      file_key = file_key[:-2]
    group = self.lm_file[file_key]
    num_sentences = len(list(group.keys()))
    sentences = [group[str(i)][...] for i in range(num_sentences)]
    lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
    for i, s in enumerate(sentences):
      lm_emb[i, :s.shape[0], :, :] = s
    return lm_emb

  def tensorize_example(self, example, is_training):
    ners = example["ners"]
    sentences = example["sentences"]

    max_sentence_length = max(len(s) for s in sentences)
    max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
    text_len = np.array([len(s) for s in sentences])
    tokens = [[""] * max_sentence_length for _ in sentences]
    char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
    context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings_size])
    lemmas = []
    if "lemmas" in example:
      lemmas = example["lemmas"]
    for i, sentence in enumerate(sentences):
      for j, word in enumerate(sentence):
        tokens[i][j] = word
        if self.context_embeddings.is_in_embeddings(word):
          context_word_emb[i, j] = self.context_embeddings[word]
        elif lemmas and self.context_embeddings.is_in_embeddings(lemmas[i][j]):
          context_word_emb[i,j] = self.context_embeddings[lemmas[i][j]]
        char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]

    tokens = np.array(tokens)

    doc_key = example["doc_key"]

    lm_emb = self.load_lm_embeddings(doc_key)

    gold_labels = []
    if is_training:
      for sid, sent in enumerate(sentences):
        ner = {(s,e):self.ner_maps[t] for s,e,t in ners[sid]}
        for s in xrange(len(sent)):
          for e in xrange(s,len(sent)):
            gold_labels.append(ner.get((s,e),0))
    gold_labels = np.array(gold_labels)

    example_tensors = (tokens, context_word_emb,lm_emb, char_index, text_len, is_training, gold_labels)

    return example_tensors

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)

  def lstm_contextualize(self, text_emb, text_len, lstm_dropout):
    num_sentences = tf.shape(text_emb)[0]

    current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]

    for layer in range(self.config["contextualization_layers"]):
      with tf.variable_scope("layer_{}".format(layer), reuse=tf.AUTO_REUSE):
        with tf.variable_scope("fw_cell"):
          cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, lstm_dropout)
        with tf.variable_scope("bw_cell"):
          cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, lstm_dropout)
        state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                                 tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
        state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                                 tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

        (fw_outputs, bw_outputs), ((_, fw_final_state), (_, bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)

        text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
        text_outputs = tf.nn.dropout(text_outputs, lstm_dropout)
        if layer > 0:
          highway_gates = tf.sigmoid(
            util.projection(text_outputs, util.shape(text_outputs, 2)))  # [num_sentences, max_sentence_length, emb]
          text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
        current_inputs = text_outputs

    return text_outputs

  def get_predictions_and_loss(self, inputs):
    tokens, context_word_emb, lm_emb, char_index, text_len, is_training, gold_labels = inputs
    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
    self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
    self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

    num_sentences = tf.shape(tokens)[0]
    max_sentence_length = tf.shape(tokens)[1]

    context_emb_list = []
    context_emb_list.append(context_word_emb)
    char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
    flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
    flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
    aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
    context_emb_list.append(aggregated_char_emb)


    lm_emb_size = util.shape(lm_emb, 2)
    lm_num_layers = util.shape(lm_emb, 3)
    with tf.variable_scope("lm_aggregation"):
      self.lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
      self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))

    flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
    flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights, 1)) # [num_sentences * max_sentence_length * emb, 1]
    aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
    aggregated_lm_emb *= self.lm_scaling
    context_emb_list.append(aggregated_lm_emb)

    context_emb = tf.concat(context_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    context_emb = tf.nn.dropout(context_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]

    candidate_scores_mask = tf.logical_and(tf.expand_dims(text_len_mask,[1]),tf.expand_dims(text_len_mask,[2])) #[num_sentence, max_sentence_length,max_sentence_length]
    sentence_ends_leq_starts = tf.tile(tf.expand_dims(tf.logical_not(tf.sequence_mask(tf.range(max_sentence_length),max_sentence_length)), 0),[num_sentences,1,1]) #[num_sentence, max_sentence_length,max_sentence_length]
    candidate_scores_mask = tf.logical_and(candidate_scores_mask,sentence_ends_leq_starts)

    flattened_candidate_scores_mask = tf.reshape(candidate_scores_mask,[-1]) #[num_sentence * max_sentence_length * max_sentence_length]


    context_outputs = self.lstm_contextualize(context_emb, text_len,self.lstm_dropout) # [num_sentence, max_sentence_length, emb]


    with tf.variable_scope("candidate_starts_ffnn"):
      candidate_starts_emb = util.projection(context_outputs,self.config["ffnn_size"]) #[num_sentences, max_sentences_length,emb]
    with tf.variable_scope("candidate_ends_ffnn"):
      candidate_ends_emb = util.projection(context_outputs,self.config["ffnn_size"]) #[num_sentences, max_sentences_length, emb]


    candidate_ner_scores = util.bilinear_classifier(candidate_starts_emb,candidate_ends_emb,self.dropout,output_size=self.num_types+1)#[num_sentence, max_sentence_length,max_sentence_length,types+1]
    candidate_ner_scores = tf.boolean_mask(tf.reshape(candidate_ner_scores,[-1,self.num_types+1]),flattened_candidate_scores_mask)


    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gold_labels, logits=candidate_ner_scores)
    loss = tf.reduce_sum(loss)


    return candidate_ner_scores, loss



  def get_pred_ner(self, sentences, span_scores, is_flat_ner):
    candidates = []
    for sid,sent in enumerate(sentences):
      for s in xrange(len(sent)):
        for e in xrange(s,len(sent)):
          candidates.append((sid,s,e))

    top_spans = [[] for _ in xrange(len(sentences))]
    for i, type in enumerate(np.argmax(span_scores,axis=1)):
      if type > 0:
        sid, s,e = candidates[i]
        top_spans[sid].append((s,e,type,span_scores[i,type]))


    top_spans = [sorted(top_span,reverse=True,key=lambda x:x[3]) for top_span in top_spans]
    sent_pred_mentions = [[] for _ in xrange(len(sentences))]
    for sid, top_span in enumerate(top_spans):
      for ns,ne,t,_ in top_span:
        for ts,te,_ in sent_pred_mentions[sid]:
          if ns < ts <= ne < te or ts < ns <= te < ne:
            #for both nested and flat ner no clash is allowed
            break
          if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
            #for flat ner nested mentions are not allowed
            break
        else:
          sent_pred_mentions[sid].append((ns,ne,t))
    pred_mentions = set((sid,s,e,t) for sid, spr in enumerate(sent_pred_mentions) for s,e,t in spr)
    return pred_mentions

  def load_eval_data(self):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example

      with open(self.config["eval_path"]) as f:
        self.eval_data = [load_line(l) for l in f.readlines()]

      print("Loaded {} eval examples.".format(len(self.eval_data)))



  def evaluate(self, session, is_final_test=False):
    self.load_eval_data()

    tp,fn,fp = 0,0,0
    start_time = time.time()
    num_words = 0
    sub_tp,sub_fn,sub_fp = [0] * self.num_types,[0]*self.num_types, [0]*self.num_types

    is_flat_ner = 'flat_ner' in self.config and self.config['flat_ner']

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      candidate_ner_scores = session.run(self.predictions, feed_dict=feed_dict)

      num_words += sum(len(tok) for tok in example["sentences"])


      gold_ners = set([(sid,s,e, self.ner_maps[t]) for sid, ner in enumerate(example['ners']) for s,e,t in ner])
      pred_ners = self.get_pred_ner(example["sentences"], candidate_ner_scores,is_flat_ner)

      tp += len(gold_ners & pred_ners)
      fn += len(gold_ners - pred_ners)
      fp += len(pred_ners - gold_ners)

      if is_final_test:
        for i in xrange(self.num_types):
          sub_gm = set((sid,s,e) for sid,s,e,t in gold_ners if t ==i+1)
          sub_pm = set((sid,s,e) for sid,s,e,t in pred_ners if t == i+1)
          sub_tp[i] += len(sub_gm & sub_pm)
          sub_fn[i] += len(sub_gm - sub_pm)
          sub_fp[i] += len(sub_pm - sub_gm)


      if example_num % 10 == 0:
        print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

    used_time = time.time() - start_time
    print("Time used: %d second, %.2f w/s " % (used_time, num_words*1.0/used_time))

    m_r = 0 if tp == 0 else float(tp)/(tp+fn)
    m_p = 0 if tp == 0 else float(tp)/(tp+fp)
    m_f1 = 0 if m_p == 0 else 2.0*m_r*m_p/(m_r+m_p)

    print("Mention F1: {:.2f}%".format(m_f1*100))
    print("Mention recall: {:.2f}%".format(m_r*100))
    print("Mention precision: {:.2f}%".format(m_p*100))

    if is_final_test:
      print("****************SUB NER TYPES********************")
      for i in xrange(self.num_types):
        sub_r = 0 if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fn[i])
        sub_p = 0 if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fp[i])
        sub_f1 = 0 if sub_p == 0 else 2.0 * sub_r * sub_p / (sub_r + sub_p)

        print("{} F1: {:.2f}%".format(self.ner_types[i],sub_f1 * 100))
        print("{} recall: {:.2f}%".format(self.ner_types[i],sub_r * 100))
        print("{} precision: {:.2f}%".format(self.ner_types[i],sub_p * 100))

    summary_dict = {}
    summary_dict["Mention F1"] = m_f1
    summary_dict["Mention recall"] = m_r
    summary_dict["Mention precision"] = m_p

    return util.make_summary(summary_dict), m_f1
