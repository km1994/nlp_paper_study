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
from functools import partial
from src.utils.registry import register
from . import Linear

registry = {}
register = partial(register, registry=registry)


@register('simple')
class Prediction(nn.Module):
    def __init__(self, args, inp_features=2):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(args.hidden_size * inp_features, args.hidden_size, activations=True),
            nn.Dropout(args.dropout),
            Linear(args.hidden_size, args.num_classes),
        )

    def forward(self, a, b):
        return self.dense(torch.cat([a, b], dim=-1))


@register('full')
class AdvancedPrediction(Prediction):
    def __init__(self, args):
        super().__init__(args, inp_features=4)

    def forward(self, a, b):
        return self.dense(torch.cat([a, b, a - b, a * b], dim=-1))


@register('symmetric')
class SymmetricPrediction(AdvancedPrediction):
    def forward(self, a, b):
        return self.dense(torch.cat([a, b, (a - b).abs(), a * b], dim=-1))
