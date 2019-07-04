# -*- coding: utf-8 -*-

data_dic = {
    'ranking_data': {
        'data_root': 'C:/Users/LawLi/PycharmProjects/dialogResearch/dialog/dataset/data',
        'vocab_size': 9185,
        'num_class': 4
    }
}


class DefaultConfig(object):
    DP = 25
    DC = 500
    N = 123
    NP = 123
    NR = 19
    KP = 0.6
    K = 3
    LR = 0.03
    BATCH_SIZE = 32
    epochs = 100
    use_gpu = True


def parse(self, kwargs):
    '''
    user can update the default hyperparamter
    '''
    for k in kwargs.keys():
        if not hasattr(self, k):
            raise Exception('opt has No key: {}'.format(k))
        setattr(self, k, kwargs[k])

    data_list = ['data_root', 'w2v_path', 'rel_num', 'vocab_size']
    for r in data_list:
        setattr(self, r, data_dic[self.data][r])

    if self.model.startswith('PCNN'):
        setattr(self, 'rel_dim', 3 * self.filters_num * len(self.filters))

    print('*************************************************')
    print('user config:')
    for k in self.__class__.__dict__.keys():
        if not k.startswith('__'):
            print("{} => {}".format(k, getattr(self, k)))

    print('*************************************************')


DefaultConfig.parse = parse
opt = DefaultConfig()
