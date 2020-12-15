from collections import defaultdict

import numpy as np

import tokenization


class Example(object):
    def __init__(self, doc_key, tokens, sentence_tokens,  document_index,
                 offset=0,bert_to_orig_map = None):
        self.doc_key = doc_key
        self.tokens = tokens
        self.sentence_tokens = sentence_tokens
        self.document_index = document_index
        self.offset = offset
        self.bert_to_orig_map = bert_to_orig_map


    def bertify(self, tokenizer):
        assert self.offset == 0

        bert_tokens = []
        orig_to_bert_map = []
        orig_to_bert_end_map = []
        for t in self.tokens:
            bert_t = tokenizer.tokenize(t)
            orig_to_bert_map.append(len(bert_tokens))
            orig_to_bert_end_map.append(len(bert_tokens) + len(bert_t) - 1)
            bert_tokens.extend(bert_t)

        bert_sentence_tokens = [tokenizer.tokenize(' '.join(s)) for s in self.sentence_tokens]

        bert_to_orig_map = [-1] * len(bert_tokens)
        for i, bert_i in enumerate(orig_to_bert_map):
            bert_to_orig_map[bert_i] = i



        return Example(self.doc_key, bert_tokens, bert_sentence_tokens, self.document_index, bert_to_orig_map=bert_to_orig_map)

    def unravel_token_index(self, token_index):
        prev_sentences_len = 0
        for i, s in enumerate(self.sentence_tokens):
            if token_index < prev_sentences_len + len(s):
                token_index_in_sentence = token_index - prev_sentences_len
                return i, token_index_in_sentence
            prev_sentences_len += len(s)

        raise ValueError('token_index is out of range ({} >= {})', token_index, len(self.tokens))




def process_example(example, index):
    sentences = example["sentences"]
    sentence_tokens = [[tokenization.convert_to_unicode(w) for w in s] for s in sentences]
    tokens = sum(sentence_tokens, [])
    doc_key = example["doc_key"]

    return Example(doc_key, tokens, sentence_tokens, index)
