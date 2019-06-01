import os
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class BaseModel(object):

  @classmethod
  def set_saver(cls, save_dir):
    '''
    Args:
      save_dir: relative path to FLAGS.logdir
    '''
    # shared between train and valid model instance
    cls.saver = tf.train.Saver(var_list=None)
    cls.save_dir = os.path.join(FLAGS.logdir, save_dir)
    cls.save_path = os.path.join(cls.save_dir, "model.ckpt")

  @classmethod
  def restore(cls, session):
    ckpt = tf.train.get_checkpoint_state(cls.save_dir)
    cls.saver.restore(session, ckpt.model_checkpoint_path)

  @classmethod
  def save(cls, session, global_step):
    cls.saver.save(session, cls.save_path, global_step)
  