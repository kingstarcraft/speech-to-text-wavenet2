import tensorflow as tf


def create(filepath, batch_size=1, repeat=False, buffsize=100000):
  def _parse(record):
    keys_to_features = {
      'uid': tf.FixedLenFeature([], tf.string),
      'audio/data': tf.VarLenFeature(tf.float32),
      'audio/shape': tf.VarLenFeature(tf.int64),
      'text': tf.VarLenFeature(tf.int64)
    }
    features = tf.parse_single_example(
      record,
      features=keys_to_features
    )
    audio = features['audio/data'].values
    shape = features['audio/shape'].values
    audio = tf.reshape(audio, shape)
    audio = tf.contrib.layers.dense_to_sparse(audio)
    text = tf.contrib.layers.dense_to_sparse(features['text'])
    return audio, text, shape[0], features['uid']

  dataset = tf.data.TFRecordDataset(filepath).map(_parse).batch(batch_size=batch_size).shuffle(buffer_size=buffsize)
  if repeat:
    dataset = dataset.repeat()
  return dataset.make_one_shot_iterator().get_next()
