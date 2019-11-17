# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
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


import os
import random
import msgpack
from .utils.vocab import Vocab, Indexer
from .utils.loader import load_data, load_embeddings


class Interface:
    def __init__(self, args, log=None):
        self.args = args
        # build/load vocab and target map
        vocab_file = os.path.join(args.output_dir, 'vocab.txt')
        target_map_file = os.path.join(args.output_dir, 'target_map.txt')
        if not os.path.exists(vocab_file):
            data = load_data(self.args.data_dir)
            self.target_map = Indexer.build((sample['target'] for sample in data), log=log)
            self.target_map.save(target_map_file)
            self.vocab = Vocab.build((word for sample in data
                                      for text in (sample['text1'], sample['text2'])
                                      for word in text.split()[:self.args.max_len]),
                                     lower=args.lower_case, min_df=self.args.min_df, log=log,
                                     pretrained_embeddings=args.pretrained_embeddings,
                                     dump_filtered=os.path.join(args.output_dir, 'filtered_words.txt'))
            self.vocab.save(vocab_file)

        else:
            self.target_map = Indexer.load(target_map_file)
            self.vocab = Vocab.load(vocab_file)
        args.num_classes = len(self.target_map)
        args.num_vocab = len(self.vocab)
        args.padding = Vocab.pad()

    def load_embeddings(self):
        """generate embeddings suited for the current vocab or load previously cached ones."""
        assert self.args.pretrained_embeddings
        embedding_file = os.path.join(self.args.output_dir, 'embedding.msgpack')
        if not os.path.exists(embedding_file):
            embeddings = load_embeddings(self.args.pretrained_embeddings, self.vocab,
                                         self.args.embedding_dim, mode=self.args.embedding_mode,
                                         lower=self.args.lower_case)
            with open(embedding_file, 'wb') as f:
                msgpack.dump(embeddings, f)
        else:
            with open(embedding_file, 'rb') as f:
                embeddings = msgpack.load(f)
        return embeddings

    def pre_process(self, data, training=True):
        result = [self.process_sample(sample) for sample in data]
        if training:
            result = list(filter(lambda x: len(x['text1']) < self.args.max_len and len(x['text2']) < self.args.max_len,
                                 result))
            if not self.args.sort_by_len:
                return result
            result = sorted(result, key=lambda x: (len(x['text1']), len(x['text2']), x['text1']))
        batch_size = self.args.batch_size
        return [self.make_batch(result[i:i + batch_size]) for i in range(0, len(data), batch_size)]

    def process_sample(self, sample, with_target=True):
        text1 = sample['text1']
        text2 = sample['text2']
        if self.args.lower_case:
            text1 = text1.lower()
            text2 = text2.lower()
        processed = {
            'text1': [self.vocab.index(w) for w in text1.split()[:self.args.max_len]],
            'text2': [self.vocab.index(w) for w in text2.split()[:self.args.max_len]],
        }
        if 'target' in sample and with_target:
            target = sample['target']
            assert target in self.target_map
            processed['target'] = self.target_map.index(target)
        return processed

    def shuffle_batch(self, data):
        data = random.sample(data, len(data))
        if self.args.sort_by_len:
            return data
        batch_size = self.args.batch_size
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return list(map(self.make_batch, batches))

    def make_batch(self, batch, with_target=True):
        batch = {key: [sample[key] for sample in batch] for key in batch[0].keys()}
        if 'target' in batch and not with_target:
            del batch['target']
        batch = {key: self.padding(value, min_len=self.args.min_len) if key.startswith('text') else value
                 for key, value in batch.items()}
        return batch

    @staticmethod
    def padding(samples, min_len=1):
        max_len = max(max(map(len, samples)), min_len)
        batch = [sample + [Vocab.pad()] * (max_len - len(sample)) for sample in samples]
        return batch

    def post_process(self, output):
        final_prediction = []
        for prob in output:
            idx = max(range(len(prob)), key=prob.__getitem__)
            target = self.target_map[idx]
            final_prediction.append(target)
        return final_prediction
