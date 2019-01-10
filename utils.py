import glog
import tensorflow as tf
import tensorflow.contrib.slim as slim
import librosa
import numpy as np
import string
from tensorflow.python import pywrap_tensorflow


def restore_from_pretrain(ckpt_dir):
  def get_variable_dict():
    def conv_dict(key_prefix, value_prefix):
      reval = {}
      for buff in [('/weights', '/W'),
                   ('/BatchNorm/beta', '/beta'),
                   ('/BatchNorm/gamma', '/gamma'),
                   ('/BatchNorm/moving_mean', '/mean'),
                   ('/BatchNorm/moving_variance', '/variance')]:
        reval[key_prefix + buff[0]] = value_prefix + buff[1]
      return list(reval.items())

    def input_dict():
      return conv_dict('wavenet/input/conv', 'front/conv_in')

    def resnet_dict():
      reval = []
      for layer in range(3):
        for rate in [1, 2, 4, 8, 16]:
          block = "/block_%d_%d" % (layer, rate)
          for buff in [('/filter', '/conv_filter'), ('/gate', '/conv_gate'), ('/conv', '/conv_out')]:
            key_prefix = 'wavenet/resnet' + block + buff[0]
            value_prefix = block[1:] + buff[1]
            reval += conv_dict(key_prefix, value_prefix)
      return reval

    def output_dict():
      return conv_dict('wavenet/output/conv', 'logit/conv_1') + \
             [('wavenet/output/logit/weights', 'logit/conv_2/W'), ('wavenet/output/logit/biases', 'logit/conv_2/b')]

    return dict(input_dict() + resnet_dict() + output_dict())

  variable_dict = get_variable_dict()
  to_variables = slim.get_variables_to_restore()

  reader = pywrap_tensorflow.NewCheckpointReader(tf.train.latest_checkpoint(ckpt_dir))
  from_variable_shapes = reader.get_variable_to_shape_map()

  ops_to_restore = []
  for to_variable in to_variables:
    to_name = to_variable.op.name
    if to_name in variable_dict:
      from_name = variable_dict[to_name]
      if from_variable_shapes[from_name] == to_variable.shape:
        ops_to_restore.append(tf.assign(to_variable, reader.get_tensor(from_name)))
      else:
        glog.error("Can't restore %s from %s, the shape is different." % (to_name, from_name))
    else:
      glog.warning("Can't restore %s. the variable is not exit in %s." % (to_name, ckpt_dir))
  return ops_to_restore


class Data:
  channels = 20
  class_names = ['<EMP>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def read_wave(filepath, sample=3):
  wave, sr = librosa.load(filepath, mono=True, sr=None)
  wave = wave[::sample]
  mfcc = np.transpose(librosa.feature.mfcc(wave, sr=16000, n_mfcc=Data.channels), [1, 0])
  return mfcc


def read_txt(filepath):
  txt = open(filepath).read()
  txt = ' '.join(txt.split())
  txt = txt.translate(string.punctuation).lower()
  reval = []
  for ch in txt:
    try:
      if ch in Data.class_names:
        reval.append(Data.class_names.index(ch))
    except KeyError:
      pass
  return reval


def cvt_np2string(inputs):
  outputs = []
  for input in inputs:
    output = ''
    for i in input:
      ch = i.decode('utf-8')
      if ch == '<EMP>':
        continue
      output += i.decode('utf-8')
    outputs.append(output)
  return outputs


def _find_best_match(inputs):
  matches = []
  for input in inputs:
    for i in input:
      for match in matches:
        if i > match[-1]:
          matches.append(match + [i])
      matches.append([i])
  if len(matches) == 0:
    return matches
  else:
    return sorted(matches, key=lambda iter: len(iter), reverse=True)[0]


def _normalize(inputs):
  inputs = inputs.split(' ')
  outputs = []
  for input in inputs:
    if input != '':
      outputs.append(input)
  return outputs


def evalute(predicts, labels):
  predicts = _normalize(predicts)
  labels = _normalize(labels)
  matches = []
  for label in labels:
    match = []
    for j, predict in enumerate(predicts):
      if label == predict:
        match.append(j)
    if len(match) > 0:
      matches.append(match)
  match = _find_best_match(matches)
  return len(match), len(predicts), len(labels)


def evalutes(predicts, labels):
  size = min(len(predicts), len(labels))
  tp = 0
  pred = 0
  pos = 0
  for i in range(size):
    data = evalute(predicts[i], labels[i])
    tp += data[0]
    pred += data[1]
    pos += data[2]
  return tp, pred, pos
