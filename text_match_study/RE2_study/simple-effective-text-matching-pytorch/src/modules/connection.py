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
from . import Linear
from functools import partial
from src.utils.registry import register
registry = {}
register = partial(register, registry=registry)


@register('none')
class NullConnection(nn.Module):
    def forward(self, x, _, __):
        return x


@register('residual')
class Residual(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear = Linear(args.embedding_dim, args.hidden_size)

    def forward(self, x, res, i):
        if i == 1:
            res = self.linear(res)
        return (x + res) * math.sqrt(0.5)


@register('aug')
class AugmentedResidual(nn.Module):
    def forward(self, x, res, i):
        if i == 1:
            return torch.cat([x, res], dim=-1)  # res is embedding
        hidden_size = x.size(-1)
        x = (res[:, :, :hidden_size] + x) * math.sqrt(0.5)
        return torch.cat([x, res[:, :, hidden_size:]], dim=-1)  # latter half of res is embedding
