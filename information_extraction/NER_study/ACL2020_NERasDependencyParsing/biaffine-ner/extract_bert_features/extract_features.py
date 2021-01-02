# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import os

import h5py
import json
from tqdm import tqdm

import modeling
import tokenization
import tensorflow as tf
import numpy as np

from data import process_example

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "window_size", 511,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "stride", 127,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


def input_fn_builder(examples, window_size, stride, tokenizer):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        d = tf.data.Dataset.from_generator(
            functools.partial(convert_examples_to_features,
                              examples=examples,
                              window_size=window_size,
                              stride=stride,
                              tokenizer=tokenizer),
            dict(unique_ids=tf.int32,
                 input_ids=tf.int32,
                 input_mask=tf.int32,
                 input_type_ids=tf.int32,
                 extract_indices=tf.int32),
            dict(unique_ids=tf.TensorShape([]),
                 input_ids=tf.TensorShape([window_size]),
                 input_mask=tf.TensorShape([window_size]),
                 input_type_ids=tf.TensorShape([window_size]),
                 extract_indices=tf.TensorShape([window_size])))

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]
        extract_indices = features["extract_indices"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {
            "unique_ids": unique_ids,
            "extract_indices": extract_indices
        }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def _convert_example_to_features(example, window_start, window_end, tokens_ids_to_extract, tokenizer, seq_length):
    window_tokens = example.tokens[window_start:window_end]

    tokens = []
    input_type_ids = []
    for token in window_tokens:
        tokens.append(token)
        input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    extract_indices = [-1] * seq_length
    for i in tokens_ids_to_extract:
        assert i - window_start >= 0
        extract_indices[i - window_start] = i

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return dict(unique_ids=example.document_index,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                extract_indices=extract_indices)


def convert_examples_to_features(examples, window_size, stride, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    assert window_size % 2 == 1
    assert stride % 2 == 1

    for example in examples:
        for i in range(0, len(example.tokens), stride):
            window_center = i + window_size // 2
            token_ids_to_extract = []
            extract_start = int(np.clip(window_center - stride // 2, 0, len(example.tokens)))
            extract_end = int(np.clip(window_center + stride // 2 + 1, extract_start, len(example.tokens)))

            if i == 0:
                token_ids_to_extract.extend(range(extract_start))

            token_ids_to_extract.extend(range(extract_start, extract_end))

            if i + stride >= len(example.tokens):
                token_ids_to_extract.extend(range(extract_end, len(example.tokens)))

            token_ids_to_extract = [t for t in token_ids_to_extract if example.bert_to_orig_map[t] >= 0]

            yield _convert_example_to_features(example,
                                               i,
                                               min(i + window_size, len(example.tokens)),
                                               token_ids_to_extract,
                                               tokenizer,
                                               window_size)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=FLAGS.master,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    json_examples = []
    for file in FLAGS.input_file.split(';'):
        with open(file) as f:
            json_examples.extend((json.loads(jsonline) for jsonline in f.readlines()))

    orig_examples = []
    bert_examples = []
    for i, json_e in enumerate(json_examples):
        e = process_example(json_e, i)
        orig_examples.append(e)
        bert_examples.append(e.bertify(tokenizer))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        layer_indexes=layer_indexes,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings,
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=FLAGS.batch_size)

    input_fn = input_fn_builder(
        examples=bert_examples, window_size=FLAGS.window_size, stride=FLAGS.stride, tokenizer=tokenizer)

    writer = h5py.File(FLAGS.output_file, 'w')
    with tqdm(total=sum(len(e.tokens) for e in orig_examples)) as t:
        for result in estimator.predict(input_fn, yield_single_examples=True):
            document_index = int(result["unique_ids"])
            bert_example = bert_examples[document_index]
            orig_example = orig_examples[document_index]
            file_key = bert_example.doc_key.replace('/', ':')

            t.update(n=(result['extract_indices'] >= 0).sum())

            for output_index, bert_token_index in enumerate(result['extract_indices']):
                if bert_token_index < 0:
                    continue

                token_index = bert_example.bert_to_orig_map[bert_token_index]
                sentence_index, token_index = orig_example.unravel_token_index(token_index)

                dataset_key ="{}/{}".format(file_key, sentence_index)
                if dataset_key not in writer:
                    writer.create_dataset(dataset_key,
                                          (len(orig_example.sentence_tokens[sentence_index]), bert_config.hidden_size, len(layer_indexes)),
                                          dtype=np.float32)

                dset = writer[dataset_key]
                for j, layer_index in enumerate(layer_indexes):
                    layer_output = result["layer_output_%d" % j]
                    dset[token_index, :, j] = layer_output[output_index]
    writer.close()


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_file")
    tf.app.run()
