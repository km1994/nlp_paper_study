import numpy as np
from tqdm import tqdm


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, gaz, char_alphabet, label_alphabet, gaz_alphabet, number_normalized, max_sent_length):
    in_lines = open(input_file, 'r',encoding="utf-8").readlines()
    instance_texts = []
    instance_ids = []
    chars = []
    labels = []
    char_ids = []
    label_ids = []
    cut_num = 0
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            char = pairs[0]
            if number_normalized:
                char = normalize_word(char)
            label = pairs[-1]
            chars.append(char)
            labels.append(label)
            char_ids.append(char_alphabet.get_index(char))
            label_ids.append(label_alphabet.get_index(label))
        else:
            if ((max_sent_length < 0) or (len(chars) < max_sent_length)) and (len(chars) > 0):
                gazs = []
                gaz_ids = []
                s_length = len(chars)
                for idx in range(s_length):
                    matched_list = gaz.enumerateMatchList(chars[idx:])
                    matched_length = [len(a) for a in matched_list]
                    gazs.append(matched_list)
                    matched_id = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    if matched_id:
                        gaz_ids.append([matched_id, matched_length])
                    else:
                        gaz_ids.append([])
                instance_texts.append([chars, gazs, labels])
                instance_ids.append([char_ids, gaz_ids, label_ids])
            elif len(chars) < max_sent_length:
                cut_num += 1
            chars = []
            labels = []
            char_ids = []
            label_ids = []
            gazs = []
            gaz_ids = []
    return instance_texts, instance_ids, cut_num


def build_pretrain_embedding(embedding_path, alphabet, skip_first_row=False, separator=" ", embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path, skip_first_row, separator)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for alph, index in alphabet.iteritems():
        if alph in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph])
            else:
                pretrain_emb[index, :] = embedd_dict[alph]
            perfect_match += 1
        elif alph.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[alph.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding: %s\n     pretrain num:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
    embedding_path, pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path, skip_first_row=False, separator=" "):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r',encoding="utf-8") as file:
        i = 0
        j = 0
        for line in file:
            if i == 0:
                i = i + 1
                if skip_first_row:
                    _ = line.strip()
                    continue
            j = j+1
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(separator)
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if embedd_dim + 1 == len(tokens):
                    embedd = np.empty([1, embedd_dim])
                    embedd[:] = tokens[1:]
                    embedd_dict[tokens[0]] = embedd
                else:
                    continue
    return embedd_dict, embedd_dim
