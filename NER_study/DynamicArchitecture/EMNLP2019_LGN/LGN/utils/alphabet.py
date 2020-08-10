# -*- coding: utf-8 -*-
# @Author: Max
# @Last Modified by:   Yicheng Zou,     Contact: yczou18@fudan.edu.cn

"""
Alphabet maps objects to integer ids. It provides two way mapping from the index to the objects.
"""
import json
import os


class Alphabet:
    def __init__(self, name, label=False, keep_growing=True):
        self.__name = name
        self.UNKNOWN = "</unk>"
        self.label = label
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = 0
        self.next_index = 1
        if not self.label:
            self.add(self.UNKNOWN)

    def clear(self, keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = 0
        self.next_index = 1
        
    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]

    def get_instance(self, index):
        if index == 0:
            # First index is occupied by the wildcard element.
            return None
        try:
            return self.instances[index - 1]
        except IndexError:
            print('WARNING:Alphabet get_instance ,unknown instance index {}, return the first label.'.format(index))
            return self.instances[0]

    def size(self):
        return len(self.instances) + 1

    def iteritems(self):
        return self.instance2index.items()

    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            print("Exception: Alphabet is not saved: " + repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
