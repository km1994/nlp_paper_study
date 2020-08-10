# MPCNN
Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Network
paper link:http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP181.pdf

论文分析： http://blog.csdn.net/liuchonge/article/details/62424805 http://blog.csdn.net/liuchonge/article/details/64128870 http://blog.csdn.net/liuchonge/article/details/64440110

glove file :http://nlp.stanford.edu/data/glove.6B.zip
experiment on python3.5 and tensorflow-gpu1.4

引用代码：https://github.com/lc222/MPCNN-sentence-similarity-tensorflow

关于定位loss NAN的问题：
  1.用tfdbg命令查找到计算欧式距离的时候有些输出为0，导致最后计算loss的时候输出为NAN。
  2.利用tensorboard可视化每个层的输出及权重
 
如何解决loss NAN的问题：
  1.调低学习率
  2.梯度检验：手工计算的梯度和框架计算的梯度比较
  3.如果cost function有log 函数，tf.clip_by_value(y,1e-4)将输入为0的去掉
  4.梯度截断，效果不是很好

由于原来博主的代码存在loss NAN的问题，所以我对博主的代码做了以下修改：
  1.计算相似度层中去掉了欧氏距离或者去掉tf.sqrt函数
  2.每一卷积层加BN
  3.所有可训练的变量加入到L2正则化中
  4.activate function 都换成了 tanh
  
仍存在的问题：
  1.加上attention layer仍然会出现loss NAN的问题

未实现的想法：
  1.将欧式距离换为标准化欧氏距离
  2.dropout设为0.8-0.9会不会更容易收敛
  
train.png是训练集的acc和loss曲线
valid.png是验证集的acc和loss曲线
