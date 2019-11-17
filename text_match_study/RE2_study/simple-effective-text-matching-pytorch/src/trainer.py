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
import random
import json5
import torch
from datetime import datetime
from pprint import pformat
from .utils.loader import load_data
from .utils.logger import Logger
from .utils.params import validate_params
from .model import Model
from .interface import Interface


class Trainer:
    def __init__(self, args):
        self.args = args
        self.log = Logger(self.args)

    def train(self):
        start_time = datetime.now()
        model, interface, states = self.build_model()
        train = load_data(self.args.data_dir, 'train')
        dev = load_data(self.args.data_dir, self.args.eval_file)
        self.log(f'train ({len(train)}) | {self.args.eval_file} ({len(dev)})')
        train_batches = interface.pre_process(train)
        dev_batches = interface.pre_process(dev, training=False)
        self.log('setup complete: {}s.'.format(str(datetime.now() - start_time).split(".")[0]))

        try:
            for epoch in range(states['start_epoch'], self.args.epochs + 1):
                states['epoch'] = epoch
                self.log.set_epoch(epoch)

                batches = interface.shuffle_batch(train_batches)
                for batch_id, batch in enumerate(batches):
                    stats = model.update(batch)
                    self.log.update(stats)
                    eval_per_updates = self.args.eval_per_updates \
                        if model.updates > self.args.eval_warmup_steps else self.args.eval_per_updates_warmup
                    if model.updates % eval_per_updates == 0 or (self.args.eval_epoch and batch_id + 1 == len(batches)):
                        self.log.newline()
                        score, dev_stats = model.evaluate(dev_batches)
                        if score > states['best_eval']:
                            states['best_eval'], states['best_epoch'], states['best_step'] = score, epoch, model.updates
                            if self.args.save:
                                model.save(states, name=model.best_model_name)
                        self.log.log_eval(dev_stats)
                        if self.args.save_all:
                            model.save(states)
                            model.save(states, name='last')
                        if model.updates - states['best_step'] > self.args.early_stopping \
                                and model.updates > self.args.min_steps:
                            self.log('[Tolerance reached. Training is stopped early.]')
                            raise EarlyStop('[Tolerance reached. Training is stopped early.]')
                    if stats['loss'] > self.args.max_loss:
                        raise EarlyStop('[Loss exceeds tolerance. Unstable training is stopped early.]')
                    if stats['lr'] < self.args.min_lr - 1e-6:
                        raise EarlyStop('[Learning rate has decayed below min_lr. Training is stopped early.]')
                self.log.newline()
            self.log('Training complete.')
        except KeyboardInterrupt:
            self.log.newline()
            self.log(f'Training interrupted. Stopped early.')
        except EarlyStop as e:
            self.log.newline()
            self.log(str(e))
        self.log(f'best dev score {states["best_eval"]} at step {states["best_step"]} '
                 f'(epoch {states["best_epoch"]}).')
        self.log(f'best eval stats [{self.log.best_eval_str}]')
        training_time = str(datetime.now() - start_time).split('.')[0]
        self.log(f'Training time: {training_time}.')
        states['start_time'] = str(start_time).split('.')[0]
        states['training_time'] = training_time
        return states

    def build_model(self):
        states = {}
        interface = Interface(self.args, self.log)
        self.log(f'#classes: {self.args.num_classes}; #vocab: {self.args.num_vocab}')
        if self.args.seed:
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            if self.args.cuda:
                torch.cuda.manual_seed(self.args.seed)
            if self.args.deterministic:
                torch.backends.cudnn.deterministic = True

        model = Model(self.args)
        print(model)
        if self.args.pretrained_embeddings:
            embeddings = interface.load_embeddings()
            model.set_embeddings(embeddings)

        # set initial states
        states['start_epoch'] = 1
        states['best_eval'] = 0.
        states['best_epoch'] = 0
        states['best_step'] = 0

        self.log(f'trainable params: {model.num_parameters():,d}')
        self.log(f'trainable params (exclude embeddings): {model.num_parameters(exclude_embed=True):,d}')
        validate_params(self.args)
        with open(os.path.join(self.args.summary_dir, 'args.json5'), 'w') as f:
            json5.dump(self.args.__dict__, f, indent=2)
        self.log(pformat(vars(self.args), indent=2, width=120))
        return model, interface, states


class EarlyStop(Exception):
    pass
