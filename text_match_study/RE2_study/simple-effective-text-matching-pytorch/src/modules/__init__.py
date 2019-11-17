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


from typing import Collection
import math
import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = {}

    def add_summary(self, name, val):
        if self.training:
            self.summary[name] = val.clone().detach().cpu().numpy()

    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        if self.summary:
            summary.update({base_name + name: val for name, val in self.summary.items()})
        for name, child in self.named_children():
            if hasattr(child, 'get_summary'):
                name = base_name + name
                summary.update(child.get_summary(name))
        return summary


class ModuleList(nn.ModuleList):
    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        for i, module in enumerate(self):
            if hasattr(module, 'get_summary'):
                name = base_name + str(i)
                summary.update(module.get_summary(name))
        return summary


class ModuleDict(nn.ModuleDict):
    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        for key, module in self.items():
            if hasattr(module, 'get_summary'):
                name = base_name + key
                summary.update(module.get_summary(name))
        return summary


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))


class Linear(nn.Module):
    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_sizes: Collection[int]):
        super().__init__()
        assert all(k % 2 == 1 for k in kernel_sizes), 'only support odd kernel sizes'
        assert out_channels % len(kernel_sizes) == 0, 'out channels must be dividable by kernels'
        out_channels = out_channels // len(kernel_sizes)
        convs = []
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=(kernel_size - 1) // 2)
            nn.init.normal_(conv.weight, std=math.sqrt(2. / (in_channels * kernel_size)))
            nn.init.zeros_(conv.bias)
            convs.append(nn.Sequential(nn.utils.weight_norm(conv), GeLU()))
        self.model = nn.ModuleList(convs)

    def forward(self, x):
        return torch.cat([encoder(x) for encoder in self.model], dim=-1)
