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


import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from functools import partial
from src.utils.registry import register
from . import Linear, Module

registry = {}
register = partial(register, registry=registry)


@register('identity')
class Alignment(Module):
    def __init__(self, args, __):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(args.hidden_size)))

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature

    def forward(self, a, b, mask_a, mask_b):
        attn = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).byte()
        attn.masked_fill_(~mask, -1e7)
        attn_a = f.softmax(attn, dim=1)
        attn_b = f.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        self.add_summary('temperature', self.temperature)
        self.add_summary('attention_a', attn_a)
        self.add_summary('attention_b', attn_b)
        return feature_a, feature_b


@register('linear')
class MappedAlignment(Alignment):
    def __init__(self, args, input_size):
        super().__init__(args, input_size)
        self.projection = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(input_size, args.hidden_size, activations=True),
        )

    def _attention(self, a, b):
        a = self.projection(a)
        b = self.projection(b)
        return super()._attention(a, b)
