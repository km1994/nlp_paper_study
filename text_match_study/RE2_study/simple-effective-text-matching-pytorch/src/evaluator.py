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


import os
from pprint import pprint
from .model import Model
from .interface import Interface
from .utils.loader import load_data


class Evaluator:
    def __init__(self, model_path, data_file):
        self.model_path = model_path
        self.data_file = data_file

    def evaluate(self):
        data = load_data(*os.path.split(self.data_file))
        model, checkpoint = Model.load(self.model_path)
        args = checkpoint['args']
        interface = Interface(args)
        batches = interface.pre_process(data, training=False)
        _, stats = model.evaluate(batches)
        pprint(stats)
