import tensorflow as tf
import utils
import wavenet
import glog
import os

flags = tf.flags
flags.DEFINE_string('input_path', 'data/demo.wav', 'path to wav file.')
flags.DEFINE_string('ckpt_dir', 'model/buriburisuri', 'Path to directory holding a checkpoint.')
FLAGS = flags.FLAGS


def main(_):
  utils.load(FLAGS.ckpt_dir+'/config.json')
  vocabulary = tf.constant(utils.Data.vocabulary)
  inputs = tf.placeholder(tf.float32, [1, None, utils.Data.num_channel])
  seq_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(inputs, axis=2), 0.), tf.int32), axis=1)

  logits = wavenet.bulid_wavenet(inputs, len(utils.Data.vocabulary), is_training=False)
  decodes, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_len, merge_repeated=False)
  outputs = tf.sparse.to_dense(decodes[0]) + 1
  outputs = tf.gather(vocabulary, outputs)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if os.path.exists(FLAGS.ckpt_dir) and len(os.listdir(FLAGS.ckpt_dir)) > 0:
      saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
    output = utils.cvt_np2string(sess.run(outputs, feed_dict={inputs: [utils.read_wave(FLAGS.input_path)]}))[0]
    glog.info('%s: %s.', FLAGS.input_path, output)


if __name__ == '__main__':
  tf.app.run()
