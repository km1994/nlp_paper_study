import sys
import random
from utils.alphabet import Alphabet
from utils.functions import *
from utils.gazetter import Gazetteer


class Data:
    def __init__(self):
        self.max_sentence_length = 200
        self.number_normalized = True
        self.norm_char_emb = True
        self.norm_gaz_emb = True
        self.dataset_name = 'msra'
        self.tagscheme = "NoSeg"
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', unkflag=False)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')
        self.train_ids = []
        self.dev_ids = []
        self.test_ids = []
        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.char_emb_dim = 100
        self.gaz_emb_dim = 100
        self.pretrain_char_embedding = None
        self.pretrain_gaz_embedding = None
        self.dev_cut_num = 0
        self.train_cut_num = 0
        self.test_cut_num = 0
        self.cut_num = 0

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Dataset        name: %s" % self.dataset_name)
        print("     Tag          scheme: %s" % (self.tagscheme))
        print("     Max Sentence Length: %s" % self.max_sentence_length)
        print("     Char  alphabet size: %s" % self.char_alphabet.size())
        print("     Gaz   alphabet size: %s" % self.gaz_alphabet.size())
        print("     Label alphabet size: %s" % self.label_alphabet.size())
        print("     Char embedding size: %s" % self.char_emb_dim)
        print("     Gaz embedding  size: %s" % self.gaz_emb_dim)
        print("     Number   normalized: %s" % self.number_normalized)
        print("     Norm    char    emb: %s" % self.norm_char_emb)
        print("     Norm     gaz    emb: %s" % self.norm_gaz_emb)
        print("     Train instance number: %s" % (len(self.train_ids)))
        print("     Dev   instance number: %s" % (len(self.dev_ids)))
        print("     Test  instance number: %s" % (len(self.test_ids)))
        if self.cut_num != 0:
            print("     Train&Dev  cut number: %s" % self.cut_num)
        else:
            print("     Train    cut   number: %s" % self.train_cut_num)
            print("     Dev     cut    number: %s" % self.dev_cut_num)
        print("     Test    cut    number: %s" % self.test_cut_num)
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def build_gaz_file(self, gaz_file, skip_first_row=False, separator=" "):
        ## build gaz file,initial read gaz embedding file
        if gaz_file:
            with open(gaz_file, 'r',encoding="utf-8") as file:
                i = 0
                for line in tqdm(file):
                    if i == 0:
                        i = i + 1
                        if skip_first_row:
                            _ = line.strip()
                            continue
                    fin = line.strip().split(separator)[0]
                    if fin:
                        self.gaz.insert(fin, "one_source")
            print("Load gaz file: ", gaz_file, " total size:", self.gaz.size())
        else:
            print("Gaz file is None, load nothing")

    def fix_alphabet(self):
        self.char_alphabet.close()
        self.label_alphabet.close()
        self.gaz_alphabet.close()

    def build_char_pretrain_emb(self, emb_path, skip_first_row=False, separator=" "):
        print("build char pretrain emb...")
        self.pretrain_char_embedding, self.char_emb_dim = \
            build_pretrain_embedding(emb_path, self.char_alphabet, skip_first_row, separator,
                                                                                   self.char_emb_dim,
                                                                                   self.norm_char_emb)

    def build_gaz_pretrain_emb(self, emb_path, skip_first_row=True, separator=" "):
        print("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet, skip_first_row, separator,
                                                                                 self.gaz_emb_dim,
                                                                                 self.norm_gaz_emb)

    def generate_instance(self, input_file, name, random_split=False):
        texts, ids, cut_num = read_instance(input_file, self.gaz, self.char_alphabet, self.label_alphabet, self.gaz_alphabet, self.number_normalized, self.max_sentence_length)
        if name == "train":
            if random_split:
                random.seed(1)
                ix = [i for i in range(len(ids))]
                train_ix = random.sample(ix, int(len(ids) * 0.9))
                dev_ix = list(set(ix).difference(set(train_ix)))
                self.train_ids = [ids[ele] for ele in train_ix]
                self.dev_ids = [ids[ele] for ele in dev_ix]
                self.train_texts = [texts[ele] for ele in train_ix]
                self.dev_texts = [texts[ele] for ele in dev_ix]
                self.cut_num = cut_num
            else:
                self.train_ids = ids
                self.train_texts = texts
                self.train_cut_num = cut_num
        elif name == "dev":
            self.dev_ids = ids
            self.dev_texts = texts
            self.dev_cut_num = cut_num
        elif name == "test":
            self.test_ids = ids
            self.test_texts = texts
            self.test_cut_num = cut_num
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % name)

    def get_tag_scheme(self):
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagscheme = "BMES"
            else:
                self.tagscheme = "BIO"

