from utils.trie import Trie

class Gazetteer:
    def __init__(self, lower):#lower = False
        self.trie = Trie()
        self.ent2type = {} ## word list to type
        self.ent2id = {"<UNK>":0}   ## word list to id
        self.lower = lower
        self.space = ""

    def enumerateMatchList(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        match_list = self.trie.enumerateMatch(word_list, self.space)
        return match_list

    def insert(self, word_list, source):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        self.trie.insert(word_list)
        string = self.space.join(word_list)
        if string not in self.ent2type:
            self.ent2type[string] = source
        if string not in self.ent2id:
            self.ent2id[string] = len(self.ent2id)

    def searchId(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2id:
            return self.ent2id[string]
        return self.ent2id["<UNK>"]

    def searchType(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2type:
            return self.ent2type[string]
        print("Error in finding entity type at gazetteer.py, exit program! String:", string)
        exit(0)

    def size(self):
        return len(self.ent2type)

    def clean(self):
        self.trie = Trie()
        self.ent2type = {}
        self.ent2id = {"<UNK>": 0}
        self.space = ""

