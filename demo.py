import tensorflow as tf
import utils
import wavenet
import glog
import os

flags = tf.flags
flags.DEFINE_string('input_path', 'data/demo.wav', 'path to wav file.')
flags.DEFINE_string('ckpt_path', 'release/buriburisuri', 'Path to directory holding a checkpoint.')
FLAGS = flags.FLAGS


def main(_):
  if not os.path.exists(FLAGS.ckpt_path + '.index'):
    glog.error('%s was not found.' % FLAGS.ckpt_path)
    return -1

  utils.load(FLAGS.ckpt_path + '.json')
  vocabulary = tf.constant(utils.Data.vocabulary)
  inputs = tf.placeholder(tf.float32, [1, None, utils.Data.num_channel])
  sequence_length = tf.placeholder(tf.int32, [None])

  logits = wavenet.bulid_wavenet(inputs, len(utils.Data.vocabulary), is_training=False)
  decodes, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]), sequence_length,
                                             merge_repeated=False)
  outputs = tf.gather(vocabulary, tf.sparse.to_dense(decodes[0]))
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, FLAGS.ckpt_path)
    wave = utils.read_wave(FLAGS.input_path)
    output = utils.cvt_np2string(sess.run(outputs, feed_dict={inputs: [wave], sequence_length: [wave.shape[0]]}))[0]
    glog.info('%s: %s.', FLAGS.input_path, output)
  return 0


if __name__ == '__main__':
  tf.app.run()
