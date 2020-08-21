import torch
from utils.graph_generator import *


def batchify(input_batch_list, gpu):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    gazs = [sent[1] for sent in input_batch_list]
    labels = [sent[2] for sent in input_batch_list]
    word_seq_lengths = list(map(len, words))
    max_seq_len = max(word_seq_lengths)
    gazs_list, gaz_lens, max_gaz_len = seq_gaz(gazs)
    tmp_matrix = list(map(graph_generator, [(max_gaz_len, max_seq_len, gaz) for gaz in gazs]))
    batch_t_matrix = torch.ByteTensor([ele[0] for ele in tmp_matrix])
    batch_c_matrix = torch.ByteTensor([ele[1] for ele in tmp_matrix])
    batch_l_matrix = torch.ByteTensor([ele[2] for ele in tmp_matrix])
    gazs_tensor = torch.zeros((batch_size, max_gaz_len), requires_grad=False).long()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=False).bool()
    for idx, (seq, gaz, gaz_len, label, seqlen) in enumerate(zip(words, gazs_list, gaz_lens, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        gazs_tensor[idx, :gaz_len] = torch.LongTensor(gaz)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen).bool()
    word_seq_lengths = torch.LongTensor(word_seq_lengths)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    gazs_tensor = gazs_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    batch_t_matrix = batch_t_matrix[word_perm_idx]
    batch_c_matrix = batch_c_matrix[word_perm_idx]
    batch_l_matrix = batch_l_matrix[word_perm_idx]
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
        batch_t_matrix = batch_t_matrix.cuda()
        gazs_tensor = gazs_tensor.cuda()
        batch_c_matrix = batch_c_matrix.cuda()
        batch_l_matrix = batch_l_matrix.cuda()
    return word_seq_tensor, word_seq_lengths, gazs_tensor, mask, label_seq_tensor, word_seq_recover, batch_t_matrix, batch_c_matrix, batch_l_matrix



