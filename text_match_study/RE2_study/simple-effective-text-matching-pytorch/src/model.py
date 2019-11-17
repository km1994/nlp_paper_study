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
import math
import random
import torch
import torch.nn.functional as f
from tqdm import tqdm
from .network import Network
from .utils.metrics import registry as metrics


class Model:
    prefix = 'checkpoint'
    best_model_name = 'best.pt'

    def __init__(self, args, state_dict=None):
        self.args = args

        # network
        self.network = Network(args)
        self.device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
        self.network.to(self.device)
        # optimizer
        self.params = list(filter(lambda x: x.requires_grad, self.network.parameters()))
        self.opt = torch.optim.Adam(self.params, args.lr, betas=(args.beta1, args.beta2),
                                    weight_decay=args.weight_decay)
        # updates
        self.updates = state_dict['updates'] if state_dict else 0

        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['model'].keys()):
                if k not in new_state:
                    del state_dict['model'][k]
            self.network.load_state_dict(state_dict['model'])
            self.opt.load_state_dict(state_dict['opt'])

    def _update_schedule(self):
        if self.args.lr_decay_rate < 1.:
            args = self.args
            t = self.updates
            base_ratio = args.min_lr / args.lr
            if t < args.lr_warmup_steps:
                ratio = base_ratio + (1. - base_ratio) / max(1., args.lr_warmup_steps) * t
            else:
                ratio = max(base_ratio, args.lr_decay_rate ** math.floor((t - args.lr_warmup_steps) /
                                                                         args.lr_decay_steps))
            self.opt.param_groups[0]['lr'] = args.lr * ratio

    def update(self, batch):
        self.network.train()
        self.opt.zero_grad()
        inputs, target = self.process_data(batch)
        output = self.network(inputs)
        summary = self.network.get_summary()
        loss = self.get_loss(output, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_clipping)
        assert grad_norm >= 0, 'encounter nan in gradients.'
        self.opt.step()
        self._update_schedule()
        self.updates += 1
        stats = {
            'updates': self.updates,
            'loss': loss.item(),
            'lr': self.opt.param_groups[0]['lr'],
            'gnorm': grad_norm,
            'summary': summary,
        }
        return stats

    def evaluate(self, data):
        self.network.eval()
        targets = []
        probabilities = []
        predictions = []
        losses = []
        for batch in tqdm(data[:self.args.eval_subset], desc='evaluating', leave=False):
            inputs, target = self.process_data(batch)
            with torch.no_grad():
                output = self.network(inputs)
                loss = self.get_loss(output, target)
                pred = torch.argmax(output, dim=1)
                prob = torch.nn.functional.softmax(output, dim=1)
                losses.append(loss.item())
                targets.extend(target.tolist())
                probabilities.extend(prob.tolist())
                predictions.extend(pred.tolist())
        outputs = {
            'target': targets,
            'prob': probabilities,
            'pred': predictions,
            'args': self.args,
        }
        stats = {
            'updates': self.updates,
            'loss': sum(losses[:-1]) / (len(losses) - 1) if len(losses) > 1 else sum(losses),
        }
        for metric in self.args.watch_metrics:
            if metric not in stats:  # multiple metrics could be computed by the same function
                stats.update(metrics[metric](outputs))
        assert 'score' not in stats, 'metric name collides with "score"'
        eval_score = stats[self.args.metric]
        stats['score'] = eval_score
        return eval_score, stats  # first value is for early stopping

    def predict(self, batch):
        self.network.eval()
        inputs, _ = self.process_data(batch)
        with torch.no_grad():
            output = self.network(inputs)
            output = torch.nn.functional.softmax(output, dim=1)
        return output.tolist()

    def process_data(self, batch):
        text1 = torch.LongTensor(batch['text1']).to(self.device)
        text2 = torch.LongTensor(batch['text2']).to(self.device)
        mask1 = torch.ne(text1, self.args.padding).unsqueeze(2)
        mask2 = torch.ne(text2, self.args.padding).unsqueeze(2)
        inputs = {
            'text1': text1,
            'text2': text2,
            'mask1': mask1,
            'mask2': mask2,
        }
        if 'target' in batch:
            target = torch.LongTensor(batch['target']).to(self.device)
            return inputs, target
        return inputs, None

    @staticmethod
    def get_loss(logits, target):
        return f.cross_entropy(logits, target)

    def save(self, states, name=None):
        if name:
            filename = os.path.join(self.args.summary_dir, name)
        else:
            filename = os.path.join(self.args.summary_dir, f'{self.prefix}_{self.updates}.pt')
        params = {
            'state_dict': {
                'model': self.network.state_dict(),
                'opt': self.opt.state_dict(),
                'updates': self.updates,
            },
            'args': self.args,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state()
        }
        params.update(states)
        if self.args.cuda:
            params['torch_cuda_state'] = torch.cuda.get_rng_state()
        torch.save(params, filename)

    @classmethod
    def load(cls, file):
        checkpoint = torch.load(file, map_location=(
            lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
        ))
        prev_args = checkpoint['args']
        # update args
        prev_args.output_dir = os.path.dirname(os.path.dirname(file))
        prev_args.summary_dir = os.path.join(prev_args.output_dir, prev_args.name)
        prev_args.cuda = prev_args.cuda and torch.cuda.is_available()
        return cls(prev_args, state_dict=checkpoint['state_dict']), checkpoint

    def num_parameters(self, exclude_embed=False):
        num_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        if exclude_embed:
            num_params -= 0 if self.args.fix_embeddings else next(self.network.embedding.parameters()).numel()
        return num_params

    def set_embeddings(self, embeddings):
        self.network.embedding.set_(embeddings)
