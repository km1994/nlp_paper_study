# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
import torch.nn.functional as f
from . import Conv1d


class Encoder(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()
        self.dropout = args.dropout
        self.encoders = nn.ModuleList([Conv1d(
                in_channels=input_size if i == 0 else args.hidden_size,
                out_channels=args.hidden_size,
                kernel_sizes=args.kernel_sizes) for i in range(args.enc_layers)])

    def forward(self, x, mask):
        x = x.transpose(1, 2)  # B x C x L
        mask = mask.transpose(1, 2)
        for i, encoder in enumerate(self.encoders):
            x.masked_fill_(~mask, 0.)
            if i > 0:
                x = f.dropout(x, self.dropout, self.training)
            x = encoder(x)
        x = f.dropout(x, self.dropout, self.training)
        return x.transpose(1, 2)  # B x L x C
