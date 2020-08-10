# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Last Modified by:   Yicheng Zou,     Contact: yczou18@fudan.edu.cn

import numpy as np


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance_with_gaz(input_file, word_dict, char_alphabet, word_alphabet, label_alphabet, number_normalized, max_sent_length):
    instence_texts = []
    instence_Ids = []

    with open(input_file, 'r', encoding="utf-8") as f:

        chars = []
        labels = []
        char_Ids = []
        label_Ids = []

        for line in f:
            if len(line) > 1:
                pairs = line.strip().split()
                char = pairs[0]
                if number_normalized:
                    char = normalize_word(char)
                chars.append(char)
                char_Ids.append(char_alphabet.get_index(char))
                if len(pairs) > 1:
                    label = pairs[-1]
                else:
                    label = 'O'
                labels.append(label)
                label_Ids.append(label_alphabet.get_index(label))

            # A sentence is finished.
            else:
                # Only keep the sentence whose length is smaller than MAX_SENT_LENGTH.
                if ((max_sent_length < 0) or (len(chars) < max_sent_length)) and (len(chars)>0):
                    words = []
                    word_Ids = []
                    for idx in range(len(chars)):
                        matched_list = word_dict.recursive_search(chars[idx:])
                        matched_length = [len(a) for a in matched_list]

                        words.append(matched_list)
                        matched_Id = [word_alphabet.get_index(word) for word in matched_list]
                        if matched_Id:
                            word_Ids.append([matched_Id, matched_length])
                        else:
                            word_Ids.append([])

                    instence_texts.append([chars, words, labels])
                    instence_Ids.append([char_Ids, word_Ids, label_Ids])
                chars = []
                labels = []
                char_Ids = []
                label_Ids = []

    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, norm=True, embedd_dim=50):

    def norm2one(vec):
        root_sum_square = np.sqrt(np.sum(np.square(vec)))
        return vec / root_sum_square

    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    not_match = 0
    for word, index in word_alphabet.instance2index.items():
        if word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
        elif word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, match:%s, oov:%s, oov%%:%.4f" %
          (pretrained_size, word_alphabet.size() - not_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


def load_pretrain_emb(embedding_path):
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            embedd_dim = len(tokens) - 1
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim
