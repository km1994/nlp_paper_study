#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import util,biaffine_ner_model

if __name__ == "__main__":
  config = util.initialize_from_env()

  config['eval_path'] = config['test_path']

  model = biaffine_ner_model.BiaffineNERModel(config)

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  session_config.allow_soft_placement = True
  with tf.Session(config=session_config) as session:
    model.restore(session)
    model.evaluate(session,True)
