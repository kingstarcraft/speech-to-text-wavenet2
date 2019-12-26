'''
A tool to convet model from https://github.com/buriburisuri/speech-to-text-wavenet
'''
import utils
import wavenet
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('input_dir', 'D:/speech-to-text-wavenet/model', 'Directory buriburisuri model.')
flags.DEFINE_string('output_path', 'model/buriburisuri/ckpt', 'Path to output model.')
FLAGS = flags.FLAGS


def main(_):
  inputs = tf.placeholder(tf.float32, [1, None, utils.Data.num_channel])
  wavenet.bulid_wavenet(inputs, len(utils.Data.vocabulary), is_training=False)
  restore = utils.restore_from_pretrain(FLAGS.input_dir)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(restore)
    saver.save(sess, FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
