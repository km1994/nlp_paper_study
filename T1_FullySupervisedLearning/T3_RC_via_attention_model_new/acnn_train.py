from attention.utils import *
import attention.pyt_acnn as pa
from torch.utils.data import DataLoader, dataset
import numpy as np
import torch.nn.functional as F
from attention.config import opt

device = torch.device('cuda:0') if opt.use_gpu else 'cpu'
device = 'cpu'

def gen_dataloader(data, word_dict, arg):
    tp = vectorize(data, word_dict, arg.N)
    x, y, e1, e2, e1d2, e2d1, zd, d1, d2 = tp
    y_t = torch.LongTensor(np.array(y).astype(np.int64))
    zd = np.array(zd).reshape(-1, 1)
    e1, e1d2, d1 = np.array(e1).reshape(-1, 1), np.array(e1d2).reshape(-1, 1), np.array(d1)
    e2, e2d1, d2 = np.array(e2).reshape(-1, 1), np.array(e2d1).reshape(-1, 1), np.array(d2)
    np_cat = np.concatenate((x, e1, e1d2, e2, e2d1, zd, d1, d2), 1)
    d_t = torch.from_numpy(np_cat.astype(np.int64))
    ds = dataset.TensorDataset(d_t, y_t)
    return DataLoader(ds, arg.BATCH_SIZE, True)


def data_unpack(cat_data, N):
    list_x = np.split(cat_data.numpy(), [N, N + 1, N + 2, N + 3, N + 4, N + 5, 2 * N + 5], 1)
    x = torch.from_numpy(list_x[0]).to(device)
    e1 = torch.from_numpy(list_x[1]).to(device)
    e1d2 = torch.from_numpy(list_x[2]).to(device)
    e2 = torch.from_numpy(list_x[3]).to(device)
    e2d1 = torch.from_numpy(list_x[4]).to(device)
    zd = torch.from_numpy(list_x[5]).to(device)
    d1 = torch.from_numpy(list_x[6]).to(device)
    d2 = torch.from_numpy(list_x[7]).to(device)
    return x, e1, e1d2, e2, e2d1, zd, d1, d2


def prediction(wo, rel_weight, y, all_y):
    wo_norm = F.normalize(wo)  # (bs, dc)
    wo_norm_tile = wo_norm.unsqueeze(1).repeat(1, all_y.size()[0], 1)  # (bs, nr, dc)
    ay_emb = torch.mm(all_y, rel_weight)  # (nr, dc)
    dist = torch.norm(wo_norm_tile - ay_emb, 2, 2)  # (bs, nr)
    predict = torch.min(dist, 1)[1].long()
    y = torch.max(y, 1)[1]
    correct = torch.eq(predict, y)
    return correct.sum().float() / float(correct.data.size()[0])
# def prediction(wo, rel_weight, y):
#     wo_norm = F.normalize(wo)  # (bs, dc)
#     wo_norm_tile = wo_norm.unsqueeze(1).repeat(1, y.size()[-1], 1)  # (bs, nr, dc)
#     dist = torch.norm(wo_norm_tile - rel_weight, 2, 2)  # (bs, nr)
#     predict = torch.min(dist, 1)[1].long()
#     y = torch.max(y, 1)[1]
#     correct = torch.eq(predict, y)
#     return correct.sum().float() / float(correct.data.size()[0])


def model_run(opt, dataloader, loss_func, model, all_y, optimizer=None):
    acc, loss = 0, 0
    for i, (bx_cat, by) in enumerate(dataloader):
        by = by.float().to(device)
        bin_tup = data_unpack(bx_cat, opt.N)
        # wo, rel_weight = model(bin_tup, all_y)
        wo, rel_weight = model(bin_tup)
        a = prediction(wo, rel_weight, by, all_y)
        l = loss_func(wo, rel_weight, by, all_y)
        # a = prediction(wo, rel_weight, by)
        # l = loss_func(wo, rel_weight, by)
        # print('%.2f%%, %.2f' % (a.cpu().data.numpy() * 100, l.detach().cpu().numpy()))
        acc += a.cpu().data.numpy() * 100
        loss += l.detach().cpu().numpy()
        if optimizer is not None:
            l.backward(), optimizer.step(), optimizer.zero_grad()
    return acc / i, loss / i


all_y = to_categorical([i for i in range(opt.NR)], opt.NR)
all_y = torch.from_numpy(all_y.astype(np.float)).float().to(device)
train_data = load_data('attention/train.txt', opt.NR)
eval_data = load_data('attention/test.txt', opt.NR)
word_dict = build_dict(train_data[0])

train_dataloader = gen_dataloader(train_data, word_dict, opt)
eval_dataloader = gen_dataloader(eval_data, word_dict, opt)

embed_file = 'attention/embeddings.txt'
vac_file = 'attention/words.lst'
embedding = load_embedding(embed_file, vac_file, word_dict)

model = pa.ACNN(opt, embedding).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.LR, weight_decay=0.0001)  # optimize all rnn parameters
loss_func = pa.DistanceLoss(opt.NR)

for i in range(opt.epochs):
    acc, loss = model_run(opt, train_dataloader, loss_func, model, all_y, optimizer)
    val_acc, val_loss = model_run(opt, eval_dataloader, loss_func, model, all_y)
    print('epoch: %d, t_l: %.2f, t_a: %.2f%%, v_l: %.2f, v_a: %.2f%%' % (i, loss, acc, val_loss, val_acc))
