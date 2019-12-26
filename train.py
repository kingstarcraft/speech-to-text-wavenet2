import wavenet
import utils
import dataset
import tensorflow as tf
import os
import utils

flags = tf.flags
flags.DEFINE_string('ckpt_path', 'model/v28/ckpt', 'Path to directory holding a checkpoint.')
flags.DEFINE_string('train_dir', 'F:/speech2text/wavenet/train', 'Directory to train dataset.')
flags.DEFINE_string('test_dir', 'F:/speech2text/wavenet/test', 'Directory to test dataset.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate of train.')
flags.DEFINE_integer('batch_size', 32, 'Batch size of train.')
FLAGS = flags.FLAGS


def main(_):
  inputs = tf.placeholder(shape=[None, None, 20], dtype=tf.float32)
  labels = tf.placeholder(shape=[None, None], dtype=tf.int64)
  is_training = tf.placeholder(shape=[], dtype=tf.bool)
  global_step = tf.train.get_or_create_global_step()
  logits = wavenet.bulid_wavenet(inputs, len(utils.Data.vocabulary), is_training)
  loss = tf.nn.ctc_loss(labels=labels, inputs=logits, sequence_length=10, time_major=False)
  outputs, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]), sequence_length,
                                             merge_repeated=False)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    optimize = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss, global_step=global_step)
  restore_op = utils.restore_from_pretrain(FLAGS.pretrain_dir)
  save = tf.train.Saver()
  train_dataset = dataset.create(FLAGS.train_dir)
  test_dataset = dataset.create(FLAGS.test_dir)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(restore_op)
    if len(os.listdir(FLAGS.checkpoint_dir)) > 0:
      save.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))



if __name__ == '__main__':
  tf.app.run()
