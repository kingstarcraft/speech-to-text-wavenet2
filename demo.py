import tensorflow as tf
import utils
import wavenet
import librosa
import numpy as np
import data

flags = tf.flags
flags.DEFINE_string('input_path', 'data/demo.wav', 'path to wave file.')
flags.DEFINE_string('ckpt_dir', 'pretrained', 'Directory of old ckpt.')
FLAGS = flags.FLAGS


def main(_):
  def read_wave(filepath):
    wave, _ = librosa.load(filepath, mono=True, sr=None)
    wave = wave[::3]
    mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wave, sr=16000), axis=0), [0, 2, 1])
    return mfcc

  inputs = tf.placeholder(tf.float32, [1, None, 20])
  seq_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(inputs, axis=2), 0.), tf.int32), axis=1)

  logits = wavenet.bulid_wavenet(inputs, data.vocab_size, is_training=False)
  decodes, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_len,
                                             merge_repeated=False)
  outputs = tf.sparse_to_dense(decodes[0].indices, decodes[0].dense_shape, decodes[0].values) + 1

  restore = utils.restore_from_pretrain(FLAGS.ckpt_dir)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(restore)
    data.print_index(sess.run(outputs, feed_dict={inputs: read_wave(FLAGS.input_path)}))


if __name__ == '__main__':
  tf.app.run()
