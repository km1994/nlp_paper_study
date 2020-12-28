# git 手册

## git 查看最近修改的文件

- git log --name-status 每次修改的文件列表, 显示状态
- git log --name-only 每次修改的文件列表
- git log --stat 每次修改的文件列表, 及文件修改的统计
- git whatchanged 每次修改的文件列表
- git whatchanged --stat 每次修改的文件列表, 及文件修改的统计
- git show 显示最后一次的文件改变的具体内容


## git commit 到 大文件 问题

1. 问题描述

commit 时， commit 到 100+MB 的大文件，然后 Push 的时候报错：

```
$ git push origin master
Enumerating objects: 38, done.
Counting objects: 100% (38/38), done.
Delta compression using up to 4 threads
Compressing objects: 100% (32/32), done.
Writing objects: 100% (36/36), 378.92 MiB | 4.90 MiB/s, done.
Total 36 (delta 2), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (2/2), done.
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: error: Trace: 5cb81ad3b3df265e69c746e10d053bb0a876917d76e8bd0e10139ad5c0f99f27
remote: error: See http://git.io/iEPt8g for more information.
remote: error: File data/save/cb_model/baidu_qa/2-2_500/1000_checkpoint.tar is 206.01 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File data/save/cb_model/baidu_qa/2-2_500/500_checkpoint.tar is 206.01 MB; this exceeds GitHub's file size limit of 100.00 MB
To https://github.com/km1994/seq2seqAttn.git
 ! [remote rejected] master -> master (pre-receive hook declined)
error: failed to push some refs to 'https://github.com/km1994/seq2seqAttn.git'
```
2. 解决方法

> 2.1 删除 文件

- 删除 data/save/cb_model/baidu_qa/2-2_500/1000_checkpoint.tar 文件
```
$ git filter-branch --force --index-filter 'git rm -rf --cached --ignore-unmatch data/save/cb_model/baidu_qa/2-2_500/1000_checkpoint.tar' --prune-empty --tag-name-filter cat -- --all
WARNING: git-filter-branch has a glut of gotchas generating mangled history
         rewrites.  Hit Ctrl-C before proceeding to abort, then use an
         alternative filtering tool such as 'git filter-repo'
         (https://github.com/newren/git-filter-repo/) instead.  See the
         filter-branch manual page for more details; to squelch this warning,
         set FILTER_BRANCH_SQUELCH_WARNING=1.
Proceeding with filter-branch...

Rewrite 696e45e305bf8fd236d75e3172034f6d7381d65c (1/3) (0 seconds passed, remaining 0 predictRewrite f2db92c2a70158e14c2a2f1555d04d08f266b208 (2/3) (6 seconds passed, remaining 3 predicted)    rm 'data/save/cb_model/baidu_qa/2-2_500/1000_checkpoint.tar'
Rewrite e67bafd9872d13aa41abfd2c0f560884ff34edf7 (3/3) (13 seconds passed, remaining 0 predicted)    rm 'data/save/cb_model/baidu_qa/2-2_500/1000_checkpoint.tar'

Ref 'refs/remotes/origin/master' was rewritten
WARNING: Ref 'refs/remotes/origin/master' is unchanged

W9007059@Pw9007059 MINGW64 /d/project/python_wp/nlp/NLPTask/seq2seqAttn ((23b5a24...))
```
- 删除 data/save/cb_model/baidu_qa/2-2_500/500_checkpoint.tar 文件
```
$ git filter-branch --force --index-filter 'git rm -rf --cached --ignore-unmatch data/save/cb_model/baidu_qa/2-2_500/500_checkpoint.tar' --prune-empty --tag-name-filter cat -- --all
WARNING: git-filter-branch has a glut of gotchas generating mangled history
         rewrites.  Hit Ctrl-C before proceeding to abort, then use an
         alternative filtering tool such as 'git filter-repo'
         (https://github.com/newren/git-filter-repo/) instead.  See the
         filter-branch manual page for more details; to squelch this warning,
         set FILTER_BRANCH_SQUELCH_WARNING=1.
Proceeding with filter-branch...

Rewrite 83efdc2b20382369cfa261d9c88dae06791d7ccb (1/3) (1 seconds passed, remaining 2 predictRewrite ae91e3574f84f60242d3fb20d9140dc00f7b8e63 (1/3) (1 seconds passed, remaining 2 predicted)    rm 'data/save/cb_model/baidu_qa/2-2_500/500_checkpoint.tar'
Rewrite a533826eb7c7985badd2a18023a0ebf68d0011d6 (3/3) (12 seconds passed, remaining 0 predicted)    rm 'data/save/cb_model/baidu_qa/2-2_500/500_checkpoint.tar'

Ref 'refs/heads/master' was rewritten
WARNING: Ref 'refs/remotes/origin/master' is unchanged
WARNING: Ref 'refs/remotes/origin/master' is unchanged
```

> 2.2 推送修改后的 repo（记住 这里不能 用 add 和 commit，不然会 重新 引入 大文件）
```
    $ git push origin master --force
    Enumerating objects: 32, done.
    Counting objects: 100% (32/32), done.
    Delta compression using up to 4 threads
    Compressing objects: 100% (30/30), done.
    Writing objects: 100% (32/32), 640.90 KiB | 11.24 MiB/s, done.
    Total 32 (delta 1), reused 2 (delta 0), pack-reused 0
    remote: Resolving deltas: 100% (1/1), done.
    To https://github.com/km1994/seq2seqAttn.git
    + 696e45e...ba05db6 master -> master (forced update)
```
> 2.3 清理和回收空间

虽然上面我们已经删除了文件, 但是我们的repo里面仍然保留了这些objects, 等待垃圾回收(GC), 所以我们要用命令彻底清除它, 并收回空间，命令如下:

```
    $ rm -rf .git/refs/original/
    $ git reflog expire --expire=now --all
    $ git gc --prune=now
```

3. 参考资料

1. [GitHub 上传文件过大报错：remote: error: GH001: Large files detected.](https://www.cnblogs.com/xym4869/p/11947181.html)


## git 回滚

### 查看 修改日志

```shell
$ git log
commit e67bafd9872d13aa41abfd2c0f560884ff34edf7 (HEAD -> master)
Author: unknown <W9007059@adc.com>
Date:   Sat Sep 19 18:48:18 2020 +0800

    利用 seq2seqAttn 解决 QA 问题

commit f2db92c2a70158e14c2a2f1555d04d08f266b208
Author: unknown <W9007059@adc.com>
Date:   Sat Sep 19 18:44:11 2020 +0800

    seq2seqAttn 复现”

commit 696e45e305bf8fd236d75e3172034f6d7381d65c (origin/master, origin/HEAD)
Author: km1994 <13484276267@163.com>
Date:   Wed Sep 16 18:52:34 2020 +0800

    Initial commit

W9007059@Pw9007059 MINGW64 /d/project/python_wp/nlp/NLPTask/seq2seqAttn (master)
```

### 切换到指定版本

```

$ git checkout 696e45e305bf8fd236d75e3172034f6d7381d65c
Note: switching to '696e45e305bf8fd236d75e3172034f6d7381d65c'.

You are in 'detached HEAD' state. You can look around, make experimental
changes and commit them, and you can discard any commits you make in this
state without impacting any branches by switching back to a branch.

If you want to create a new branch to retain commits you create, you may
do so (now or later) by using -c with the switch command. Example:

  git switch -c <new-branch-name>

Or undo this operation with:

  git switch -

Turn off this advice by setting config variable advice.detachedHead to false

HEAD is now at 696e45e Initial commit

W9007059@Pw9007059 MINGW64 /d/project/python_wp/nlp/NLPTask/seq2seqAttn ((696e45e...))
```

### 重新 add commit push 三联

```
$ git add .
$ git commit -m "seq2seqAttn 复现"
```
