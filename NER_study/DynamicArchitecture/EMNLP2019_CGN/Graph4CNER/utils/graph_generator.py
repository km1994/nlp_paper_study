import numpy as np


def seq_gaz(batch_gaz_ids):
    gaz_len = []
    gaz_list = []
    for gaz_id in batch_gaz_ids:
        gaz = []
        length = 0
        for ele in gaz_id:
            if ele:
                length = length + len(ele[0])
                for j in range(len(ele[0])):
                    gaz.append(ele[0][j])
        gaz_list.append(gaz)
        gaz_len.append(length)
    return gaz_list, gaz_len, max(gaz_len)


def graph_generator(input):
    max_gaz_len, max_seq_len, gaz_ids = input
    gaz_seq = []
    sentence_len = len(gaz_ids)
    gaz_len = 0
    for ele in gaz_ids:
        if ele:
            gaz_len += len(ele[0])
    matrix_size = max_gaz_len + max_seq_len
    t_matrix = np.eye(matrix_size, dtype=int)
    l_matrix = np.eye(matrix_size, dtype=int)
    c_matrix = np.eye(matrix_size, dtype=int)
    add_matrix1 = np.zeros((matrix_size, matrix_size), dtype=int)
    add_matrix2 = np.zeros((matrix_size, matrix_size), dtype=int)
    add_matrix1[:sentence_len, :sentence_len] = np.eye(sentence_len, k=1, dtype=int)
    add_matrix2[:sentence_len, :sentence_len] = np.eye(sentence_len, k=-1, dtype=int)
    t_matrix = t_matrix + add_matrix1 + add_matrix2
    l_matrix = l_matrix + add_matrix1 + add_matrix2
    # give word a index
    word_id = [[]] * sentence_len
    index = max_seq_len
    for i in range(sentence_len):
        if gaz_ids[i]:
            word_id[i] = [0] * len(gaz_ids[i][1])
            for j in range(len(gaz_ids[i][1])):
                word_id[i][j] = index
                index = index + 1
    index_gaz = max_seq_len
    index_char = 0
    for k in range(len(gaz_ids)):
        ele = gaz_ids[k]
        if ele:
            for i in range(len(ele[0])):
                gaz_seq.append(ele[0][i])
                l_matrix[index_gaz, index_char] = 1
                l_matrix[index_char, index_gaz] = 1
                l_matrix[index_gaz, index_char + ele[1][i] - 1] = 1
                l_matrix[index_char + ele[1][i] - 1, index_gaz] = 1
                for m in range(ele[1][i]):
                    c_matrix[index_gaz, index_char + m] = 1
                    c_matrix[index_char + m, index_gaz] = 1
                # char and word connection
                if index_char > 0:
                    t_matrix[index_gaz, index_char - 1] = 1
                    t_matrix[index_char - 1, index_gaz] = 1

                    if index_char + ele[1][i] < sentence_len:
                        t_matrix[index_gaz, index_char + ele[1][i]] = 1
                        t_matrix[index_char + ele[1][i], index_gaz] = 1
                else:
                    t_matrix[index_gaz, index_char + ele[1][i]] = 1
                    t_matrix[index_char + ele[1][i], index_gaz] = 1
                # word and word connection
                if index_char + ele[1][i] < sentence_len:
                    if gaz_ids[index_char + ele[1][i]]:
                        for p in range(len(gaz_ids[index_char + ele[1][i]][1])):
                            q = word_id[index_char + ele[1][i]][p]
                            t_matrix[index_gaz, q] = 1
                            t_matrix[q, index_gaz] = 1
                index_gaz = index_gaz + 1
        index_char = index_char + 1
    return (t_matrix, c_matrix, l_matrix)
