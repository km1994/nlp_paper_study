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


from collections import Counter


class Indexer:
    def __init__(self):
        self.w2id = {}
        self.id2w = {}

    @property
    def n_spec(self):
        return 0

    def __len__(self):
        return len(self.w2id)

    def __getitem__(self, index):
        if index not in self.id2w:
            raise IndexError(f'invalid index {index} in indices.')
        return self.id2w[index]

    def __contains__(self, item):
        return item in self.w2id

    def index(self, symbol):
        if symbol in self.w2id:
            return self.w2id[symbol]
        raise IndexError(f'Unknown symbol {symbol}')

    def keys(self):
        return self.w2id.keys()

    def indices(self):
        return self.id2w.keys()

    def add_symbol(self, symbol):
        if symbol not in self.w2id:
            self.id2w[len(self.id2w)] = symbol
            self.w2id[symbol] = len(self.w2id)

    @classmethod
    def build(cls, symbols, min_counts=1, dump_filtered=None, log=print):
        counter = Counter(symbols)
        symbols = sorted([t for t, c in counter.items() if c >= min_counts],
                         key=counter.get, reverse=True)
        log(f'''{len(symbols)} symbols found: {' '.join(symbols[:15]) + ('...' if len(symbols) > 15 else '')}''')
        filtered = sorted(list(counter.keys() - set(symbols)), key=counter.get, reverse=True)
        if filtered:
            log('filtered classes:')
            if len(filtered) > 20:
                log('{} ... {}'.format(' '.join(filtered[:10]), ' '.join(filtered[-10:])))
            else:
                log(' '.join(filtered))
            if dump_filtered:
                with open(dump_filtered, 'w',encoding="utf8") as f:
                    for name in filtered:
                        f.write(f'{name} {counter.get(name)}\n')
        indexer = cls()
        try:  # restore numeric order if labels are represented by integers already
            symbols = list(map(int, symbols))
            symbols.sort()
            symbols = list(map(str, symbols))
        except ValueError:
            pass
        for symbol in symbols:
            if symbol:
                indexer.add_symbol(symbol)
        return indexer

    def save(self, file):
        with open(file, 'w',encoding="utf8") as f:
            for symbol, index in self.w2id.items():
                if index < self.n_spec:
                    continue
                f.write('{}\n'.format(symbol))

    @classmethod
    def load(cls, file):
        indexer = cls()
        with open(file,encoding="utf8") as f:
            for line in f:
                symbol = line.rstrip()
                assert len(symbol) > 0, 'Empty symbol encountered.'
                indexer.add_symbol(symbol)
        return indexer


class RobustIndexer(Indexer):
    def __init__(self, validate=True):
        super().__init__()
        self.w2id.update({self.unk_symbol(): self.unk()})
        self.id2w = {i: w for w, i in self.w2id.items()}
        if validate:
            self.validate_spec()

    @property
    def n_spec(self):
        return 1

    def index(self, symbol):
        return self.w2id[symbol] if symbol in self.w2id else self.unk()

    @staticmethod
    def unk():
        return 0

    @staticmethod
    def unk_symbol():
        return '<UNK>'

    def validate_spec(self):
        assert self.n_spec == len(self.w2id), f'{self.n_spec}, {len(self.w2id)}'
        assert len(self.w2id) == max(self.id2w.keys()) + 1, "empty indices found in special tokens"
        assert len(self.w2id) == len(self.id2w), "index conflict in special tokens"


class Vocab(RobustIndexer):
    def __init__(self):
        super().__init__(validate=False)
        self.w2id.update({
            self.pad_symbol(): self.pad(),
        })
        self.id2w = {i: w for w, i in self.w2id.items()}
        self.validate_spec()

    @classmethod
    def build(cls, words, lower=False, min_df=1, max_tokens=float('inf'), pretrained_embeddings=None,
              dump_filtered=None, log=print):
        if pretrained_embeddings:
            wv_vocab = cls.load_embedding_vocab(pretrained_embeddings, lower)
        else:
            wv_vocab = set()
        if lower:
            words = (word.lower() for word in words)
        counter = Counter(words)
        candidate_tokens = sorted([t for t, c in counter.items() if t in wv_vocab or c >= min_df],
                                  key=counter.get, reverse=True)
        if len(candidate_tokens) > max_tokens:
            tokens = []
            for i, token in enumerate(candidate_tokens):
                if i < max_tokens:
                    tokens.append(token)
                elif token in wv_vocab:
                    tokens.append(token)
        else:
            tokens = candidate_tokens
        total = sum(counter.values())
        matched = sum(counter[t] for t in tokens)
        stats = (len(tokens), len(counter), total - matched, total, (total - matched) / total * 100)
        log('vocab coverage {}/{} | OOV occurrences {}/{} ({:.4f}%)'.format(*stats))
        tokens_set = set(tokens)
        if pretrained_embeddings:
            oop_samples = sorted(list(tokens_set - wv_vocab), key=counter.get, reverse=True)
            log('Covered by pretrained vectors {:.4f}%. '.format(len(tokens_set & wv_vocab) / len(tokens) * 100) +
                ('outside pretrained: ' + ' '.join(oop_samples[:10]) + ' ...' if len(oop_samples) > 10 else '')
                if oop_samples else '')
        log('top words:\n{}'.format(' '.join(tokens[:10])))
        filtered = sorted(list(counter.keys() - set(tokens)), key=counter.get, reverse=True)
        if filtered:
            if len(filtered) > 20:
                log('filtered words:\n{} ... {}'.format(' '.join(filtered[:10]), ' '.join(filtered[-10:])))
            else:
                log('filtered words:\n' + ' '.join(filtered))
            if dump_filtered:
                with open(dump_filtered, 'w',encoding="utf8") as f:
                    for name in filtered:
                        f.write(f'{name} {counter.get(name)}\n')

        vocab = cls()
        for token in tokens:
            vocab.add_symbol(token)
        return vocab

    @staticmethod
    def load_embedding_vocab(file, lower):
        wv_vocab = set()
        with open(file,encoding="utf8") as f:
            for line in f:
                token = line.rstrip().split(' ')[0]
                if lower:
                    token = token.lower()
                wv_vocab.add(token)
        return wv_vocab

    @staticmethod
    def pad():
        return 0

    @staticmethod
    def unk():
        return 1

    @property
    def n_spec(self):
        return 2

    @staticmethod
    def pad_symbol():
        return '<PAD>'

    char_map = {  # escape special characters for safe serialization
        '\n': '<NEWLINE>',
    }

    def save(self, file):
        with open(file, 'w',encoding="utf8") as f:
            for symbol, index in self.w2id.items():
                if index < self.n_spec:
                    continue
                symbol = self.char_map.get(symbol, symbol)
                f.write(f'{symbol}\n')

    @classmethod
    def load(cls, file):
        vocab = cls()
        reverse_char_map = {v: k for k, v in cls.char_map.items()}
        with open(file,encoding="utf8") as f:
            for line in f:
                symbol = line.rstrip('\n')
                symbol = reverse_char_map.get(symbol, symbol)
                vocab.add_symbol(symbol)
        return vocab
