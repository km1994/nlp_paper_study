# -*- coding: utf-8 -*-
# @Author: Yicheng Zou
# @Last Modified by:   Yicheng Zou,     Contact: yczou18@fudan.edu.cn

_end = "_end_"


class Word_Trie:
    def __init__(self):
        self.root = dict()

    def recursive_search(self, word_list):
        match_list = []
        while len(word_list) > 1:
            if self.search(word_list):
                match_list.append("".join(word_list))
            del word_list[-1]
        return match_list

    def search(self, word):
        current_dict = self.root
        for char in word:
            if char in current_dict:
                current_dict = current_dict[char]
            else:
                return False
        else:
            if _end in current_dict:
                return True
            else:
                return False

    def insert(self, word):
        current_dict = self.root
        for char in word:
            current_dict = current_dict.setdefault(char, {})
        current_dict[_end] = _end
