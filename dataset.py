import tensorflow as tf


def create(filepath, batch_size=1, repeat=False, buffsize=100000):
  def _parse(record):
    keys_to_features = {
      'uid': tf.FixedLenFeature([], tf.string),
      'wave/data': tf.VarLenFeature(tf.float32),
      'wave/shape': tf.VarLenFeature(tf.int64),
      'label': tf.VarLenFeature(tf.int64)
    }
    features = tf.parse_single_example(
      record,
      features=keys_to_features
    )
    wave_data = features['wave/data'].values
    wave_shape = features['wave/shape'].values
    wave = tf.reshape(wave_data, wave_shape)
    wave = tf.contrib.layers.dense_to_sparse(wave)
    return wave, features['label'], features['uid']

  dataset = tf.data.TFRecordDataset(filepath).map(_parse).batch(batch_size=batch_size).shuffle(buffer_size=buffsize)
  if repeat:
    dataset = dataset.repeat()
  return dataset.make_one_shot_iterator().get_next()
