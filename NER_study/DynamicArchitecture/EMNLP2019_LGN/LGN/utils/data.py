# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Last Modified by:   Yicheng Zou,     Contact: yczou18@fudan.edu.cn

import sys
from utils.alphabet import Alphabet
from utils.functions import *
from utils.word_trie import Word_Trie


class Data:
    def __init__(self): 
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_char_emb = True
        self.norm_word_emb = False
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)
        self.word_dict = Word_Trie()
        self.word_alphabet = Alphabet('word')

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []
        self.char_emb_dim = 50
        self.word_emb_dim = 50
        self.pretrain_char_embedding = None
        self.pretrain_word_embedding = None
        self.label_size = 0
        
    def show_data_summary(self):
        print("DATA SUMMARY:")
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Word  alphabet size: %s"%(self.word_alphabet.size()))
        print("     Char  alphabet size: %s"%(self.char_alphabet.size()))
        print("     Label alphabet size: %s"%(self.label_alphabet.size()))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Norm     char   emb: %s"%(self.norm_char_emb))
        print("     Norm     word   emb: %s"%(self.norm_word_emb))
        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))
        print("     Raw   instance number: %s"%(len(self.raw_texts)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def build_alphabet(self, input_file):
        self.char_alphabet.open()
        self.label_alphabet.open()

        with open(input_file, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                pair = line.split()
                char = pair[0]
                if self.number_normalized:
                    # Mapping numbers to 0
                    char = normalize_word(char)
                label = pair[-1]
                self.label_alphabet.add(label)
                self.char_alphabet.add(char)

        self.label_alphabet.close()
        self.char_alphabet.close()

    def build_word_file(self, word_file):
        # build word file,initial word embedding file
        with open(word_file, 'r', encoding="utf-8") as f:
            for line in f:
                word = line.strip().split()[0]
                if word:
                    self.word_dict.insert(word)
        print("Building the word dict...")

    def build_word_alphabet(self, input_file):
        print("Loading file: " + input_file)
        self.word_alphabet.open()
        word_list = []
        with open(input_file, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    word = line.split()[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    word_list.append(word)
                else:
                    for idx in range(len(word_list)):
                        matched_words = self.word_dict.recursive_search(word_list[idx:])
                        for matched_word in matched_words:
                            self.word_alphabet.add(matched_word)
                    word_list = []
        self.word_alphabet.close()
        print("word alphabet size:", self.word_alphabet.size())

    def build_char_pretrain_emb(self, emb_path):
        print ("Building character pretrain emb...")
        self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(emb_path, self.char_alphabet, self.norm_char_emb)

    def build_word_pretrain_emb(self, emb_path):
        print ("Building word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.norm_word_emb)

    def generate_instance_with_words(self, input_file, name):
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz(input_file, self.word_dict, self.char_alphabet,
                    self.word_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz(input_file, self.word_dict, self.char_alphabet,
                    self.word_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz(input_file, self.word_dict, self.char_alphabet,
                    self.word_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz(input_file, self.word_dict, self.char_alphabet,
                    self.word_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test/raw instance! Illegal input:%s"%(name))

    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file, 'w', encoding="utf-8")
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                # content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy] + " " + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, output_file))
