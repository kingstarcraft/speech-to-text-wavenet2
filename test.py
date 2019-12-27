import tensorflow as tf
import dataset
import wavenet
import utils
import glog
import os

flags = tf.app.flags
flags.DEFINE_string('config_path', 'config.json', 'Directory to config.')
flags.DEFINE_string('input_path', 'data/test.record', 'Path to wave file.')
flags.DEFINE_string('ckpt_dir', 'ckpt', 'Path to directory holding a checkpoint.')
FLAGS = flags.FLAGS


def main(_):
  test_dataset = dataset.create(FLAGS.test_dir, repeat=False, batch_size=1)
  waves = tf.reshape(tf.sparse.to_dense(test_dataset[0]), shape=[1, -1, utils.Data.channels])
  seq_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(waves, axis=2), 0.), tf.int32), axis=1)
  labels = tf.sparse.to_dense(test_dataset[1])
  vocabulary = tf.constant(utils.Data.vocabulary)
  labels = tf.gather(vocabulary, labels)
  logits = wavenet.bulid_wavenet(waves, len(utils.Data.vocabulary))
  decodes, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_len,
                                             merge_repeated=False)
  glob_step = tf.train.create_global_step()
  outputs = tf.sparse.to_dense(decodes[0]) + 1
  outputs = tf.gather(vocabulary, outputs)
  save = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if os.path.exists(FLAGS.checkpoint_dir) and len(os.listdir(FLAGS.checkpoint_dir)) > 0:
      save.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

    tps = 0
    preds = 0
    poses = 0
    count = 0
    glob_step = sess.run(glob_step)
    while True:
      try:
        count += 1
        y, y_ = sess.run((labels, outputs))
        y = utils.cvt_np2string(y)
        y_ = utils.cvt_np2string(y_)
        tp, pred, pos = utils.evalutes(y, y_)
        tps += tp
        preds += pred
        poses += pos
        glog.info('processed %s: tp=%d, pred=%d, pos=%d.' % (count, tp, pred, pos))
      except:
        if count % 1000 != 0:
          glog.info('processed %d.' % count)
        break
    glog.info('Evalute %d： f1=%f.' % (glob_step, 2 * tps / (preds + poses + 1e-20)))


if __name__ == '__main__':
  tf.app.run()
