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
import sys
import logging


class Logger:
    def __init__(self, args):
        log = logging.getLogger(args.summary_dir)
        if not log.handlers:
            log.setLevel(logging.DEBUG)
            fh = logging.FileHandler(os.path.join(args.summary_dir, args.log_file))
            fh.setLevel(logging.INFO)
            ch = ProgressHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            log.addHandler(fh)
            log.addHandler(ch)
        self.log = log
        # setup TensorBoard
        if args.tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(os.path.join(args.summary_dir, 'viz'))
            self.log.info(f'TensorBoard activated.')
        else:
            self.writer = None
        self.log_per_updates = args.log_per_updates
        self.summary_per_updates = args.summary_per_updates
        self.grad_clipping = args.grad_clipping
        self.clips = 0
        self.train_meters = {}
        self.epoch = None
        self.best_eval = 0.
        self.best_eval_str = ''

    def set_epoch(self, epoch):
        self(f'Epoch: {epoch}')
        self.epoch = epoch

    @staticmethod
    def _format_number(x):
        return f'{x:.4f}' if float(x) > 1e-3 else f'{x:.4e}'

    def update(self, stats):
        updates = stats.pop('updates')
        summary = stats.pop('summary')
        if updates % self.log_per_updates == 0:
            self.clips += int(stats['gnorm'] > self.grad_clipping)
            stats_str = ' '.join(f'{key}: ' + self._format_number(val) for key, val in stats.items())
            for key, val in stats.items():
                if key not in self.train_meters:
                    self.train_meters[key] = AverageMeter()
                self.train_meters[key].update(val)
            msg = f'epoch {self.epoch} updates {updates} {stats_str}'
            if self.log_per_updates != 1:
                msg = '> ' + msg
            self.log.info(msg)
            if self.writer and updates % self.summary_per_updates == 0:
                for key, val in stats.items():
                    self.writer.add_scalar(f'train/{key}', val, updates)
                for key, val in summary.items():
                    self.writer.add_histogram(key, val, updates)

    def newline(self):
        self.log.debug('')

    def log_eval(self, valid_stats):
        self.newline()
        updates = valid_stats.pop('updates')
        eval_score = valid_stats.pop('score')
        # report the exponential averaged training stats, while reporting the full dev set stats
        if self.train_meters:
            train_stats_str = ' '.join(f'{key}: ' + self._format_number(val) for key, val in self.train_meters.items())
            train_stats_str += ' ' + f'clip: {self.clips}'
            self.log.info(f'train {train_stats_str}')
        valid_stats_str = ' '.join(f'{key}: ' + self._format_number(val) for key, val in valid_stats.items())
        if eval_score > self.best_eval:
            self.best_eval_str = valid_stats_str
            self.best_eval = eval_score
            valid_stats_str += ' [NEW BEST]'
        else:
            valid_stats_str += f' [BEST: {self._format_number(self.best_eval)}]'
        self.log.info(f'valid {valid_stats_str}')
        if self.writer:
            for key in valid_stats.keys():
                group = {'valid': valid_stats[key]}
                if self.train_meters and key in self.train_meters:
                    group['train'] = float(self.train_meters[key])
                self.writer.add_scalars(f'valid/{key}', group, updates)
        self.train_meters = {}
        self.clips = 0

    def __call__(self, msg):
        self.log.info(msg)


class ProgressHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        log_entry = self.format(record)
        if record.message.startswith('> '):
            sys.stdout.write('{}\r'.format(log_entry.rstrip()))
            sys.stdout.flush()
        else:
            sys.stdout.write('{}\n'.format(log_entry))


class AverageMeter(object):
    """Keep exponential weighted averages."""
    def __init__(self, beta=0.99):
        self.beta = beta
        self.moment = 0.
        self.value = 0.
        self.t = 0.

    def update(self, val):
        self.t += 1
        self.moment = self.beta * self.moment + (1 - self.beta) * val
        # bias correction
        self.value = self.moment / (1 - self.beta ** self.t)

    def __format__(self, spec):
        return format(self.value, spec)

    def __float__(self):
        return self.value
