import glog
import tensorflow as tf
import tensorflow.contrib.slim as slim
import librosa
import numpy as np
import string
import os
import json
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
  num_channel = 20
  vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<EMP>']
  sample_rate = 16000


def load(filepath=None):
  if filepath is None:
    return
  if os.path.exists(filepath):
    data = json.load(open(filepath, encoding='utf-8'))
    if 'sample_rate' in data:
      Data.sample_rate = data['sample_rate']
    if 'num_channel' in data:
      Data.num_channel = data['num_channel']
    if 'vocabulary' in data:
      Data.vocabulary = data['vocabulary'] + ['<EMP>']
    glog.info("Load %s: sample_rate=%d, num_channel=%d, num_vocabulary=%d."
              % (filepath, Data.sample_rate, Data.num_channel, len(Data.vocabulary)))
  else:
    glog.error("Can't found %s." % filepath)


def read_wave(filepath):
  wave, sr = librosa.load(filepath, mono=True, sr=Data.sample_rate)
  mfcc = np.transpose(librosa.feature.mfcc(wave, sr=sr, n_mfcc=Data.num_channel), [1, 0])
  return mfcc


def read_txt(filepath):
  txt = open(filepath).read()
  txt = ' '.join(txt.split())
  txt = txt.translate(string.punctuation).lower()
  reval = []
  for ch in txt:
    try:
      if ch in Data.vocabulary:
        reval.append(Data.vocabulary.index(ch))
      else:
        glog.warning('%s was not in vocabulary at %s'%(ch, filepath))
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


def _find_best_match2(inputs):
  def _find_node(values, start=(-1, -1)):
    node = []
    for index in range(start[1] + 1, len(values)):
      value = values[index]
      if len(value) > 0:
        for v in value:
          if v > start[0]:
            if len(node) == 0:
              node.append((v, index))
            elif v < node[-1][0]:
              node.append((v, index))
    return node

  def _find_nodes(values):
    nodes = []
    while True:
      if len(nodes) == 0:
        node = _find_node(values)
        if len(node) == 0:
          break
        for n in node:
          nodes.append([n])
      else:
        tmps = []
        change = False
        for tmp in nodes:
          node = _find_node(values, tmp[-1])
          if len(node) == 0:
            tmps.append(tmp)
          else:
            for n in node:
              tmps.append(tmp + [n])
            change = True

        if change:
          nodes = tmps
        else:
          break
    return nodes
  nodes = _find_nodes(inputs)
  if len(nodes) == 0:
    return []
  else:
    return sorted(nodes, key=lambda iter: len(iter), reverse=True)[0]

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
  match = _find_best_match2(matches)
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
