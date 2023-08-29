#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
from pathlib import Path

from tqdm import tqdm

log = logging.getLogger(__name__)


def _fetch_from_remote(url, force_download=False, cached_dir='~/.paddle-ernie-cache'):
    import hashlib, requests, tarfile
    sig = hashlib.md5(url.encode('utf8')).hexdigest()
    cached_dir = Path(cached_dir).expanduser()
    try:
        cached_dir.mkdir()
    except OSError:
        pass
    cached_dir_model = cached_dir / sig
    if force_download or not cached_dir_model.exists():
        cached_dir_model.mkdir()
        tmpfile = cached_dir_model / 'tmp'
        with tmpfile.open('wb') as f:
            # url = 'https://ernie.bj.bcebos.com/ERNIE_stable.tgz'
            r = requests.get(url, stream=True)
            total_len = int(r.headers.get('content-length'))
            for chunk in tqdm(r.iter_content(chunk_size=1024),
                              total=total_len // 1024,
                              desc='downloading %s' % url,
                              unit='KB'):
                if chunk:
                    f.write(chunk)
                    f.flush()
            log.debug('extacting... to %s' % tmpfile)
            with tarfile.open(tmpfile.as_posix())  as tf:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tf, path=cached_dir_model.as_posix())
        os.remove(tmpfile.as_posix())
    log.debug('%s cached in %s' % (url, cached_dir))
    return cached_dir_model


def add_docstring(doc):
    def func(f):
        f.__doc__ += ('\n======other docs from supper class ======\n%s' % doc)
        return f

    return func
