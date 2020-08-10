#! /usr/bin/env python
# coding=utf-8

import re

line = "想做/ 兼_职/学生_/ 的 、加,我Q：  1 5.  8 0. ！！？？  8 6 。0.  2。 3     有,惊,喜,哦"
line = line.decode("utf8")
string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。：？?、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"), line)
print(string)
