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


import torch
import torch.nn as nn
import torch.nn.functional as f
from functools import partial
from src.utils.registry import register
from . import Linear

registry = {}
register = partial(register, registry=registry)


@register('simple')
class Fusion(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()
        self.fusion = Linear(input_size * 2, args.hidden_size, activations=True)

    def forward(self, x, align):
        return self.fusion(torch.cat([x, align], dim=-1))


@register('full')
class FullFusion(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()
        self.dropout = args.dropout
        self.fusion1 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion2 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion3 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion = Linear(args.hidden_size * 3, args.hidden_size, activations=True)

    def forward(self, x, align):
        x1 = self.fusion1(torch.cat([x, align], dim=-1))
        x2 = self.fusion2(torch.cat([x, x - align], dim=-1))
        x3 = self.fusion3(torch.cat([x, x * align], dim=-1))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = f.dropout(x, self.dropout, self.training)
        return self.fusion(x)
