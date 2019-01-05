import glog
import tensorflow as tf
import tensorflow.contrib.slim as slim
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
