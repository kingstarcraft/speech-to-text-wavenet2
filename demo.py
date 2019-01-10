import tensorflow as tf
import utils
import wavenet
import glog
import os

flags = tf.flags
flags.DEFINE_string('input_path', 'data/demo.wav', 'path to wave file.')
flags.DEFINE_string('pretrain_dir', 'pretrained', ' Directory of pretrain model.')
flags.DEFINE_string('checkpoint_dir', 'ckpt', 'Path to directory holding a checkpoint.')
FLAGS = flags.FLAGS


def main(_):
  class_names = tf.constant(utils.Data.class_names)
  inputs = tf.placeholder(tf.float32, [1, None, utils.Data.channels])
  seq_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(inputs, axis=2), 0.), tf.int32), axis=1)

  logits = wavenet.bulid_wavenet(inputs, len(utils.Data.class_names), is_training=False)
  decodes, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_len, merge_repeated=False)
  outputs = tf.sparse.to_dense(decodes[0]) + 1
  outputs = tf.gather(class_names, outputs)
  restore = utils.restore_from_pretrain(FLAGS.pretrain_dir)
  save = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(restore)
    if os.path.exists(FLAGS.checkpoint_dir) and len(os.listdir(FLAGS.checkpoint_dir)) > 0:
      save.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    output = utils.cvt_np2string(sess.run(outputs, feed_dict={inputs: [utils.read_wave(FLAGS.input_path)]}))[0]
    glog.info('%s: %s.', FLAGS.input_path, output)


if __name__ == '__main__':
  tf.app.run()
