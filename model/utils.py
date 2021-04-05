import collections
import os
import re
import six
import sys
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

from configs.constants import Constants


"""
session
"""


def make_session(target_gpu, gpu_usage):
  if target_gpu is None:
    target_gpu = str(0)
  else:
    target_gpu = str(target_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu

  if gpu_usage == 1.0:
    config = tf.ConfigProto(
      allow_soft_placement=True
    )
  elif gpu_usage == 0.0:
    config = tf.ConfigProto(
      allow_soft_placement=True,
      device_count={'GPU': 0}
    )
  else:
    config = tf.ConfigProto(
      allow_soft_placement=True,
      gpu_options=tf.GPUOptions(
        visible_device_list=target_gpu,
        per_process_gpu_memory_fraction=gpu_usage
      )
    )

  sess = tf.Session(config=config)

  return sess


"""
input iterator
"""


def get_features_from_tfrecords(tfrecord_paths, parser, is_training, flag_objs):
  if type(tfrecord_paths) is not list:
    tfrecord_paths = [tfrecord_paths]

  if is_training:
    # make parallel reading dataset
    d = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecord_paths))
    # d = d.repeat()
    d = d.shuffle(buffer_size=len(tfrecord_paths))

    #
    cycle_length = min(Constants.NUM_CPU_THREADS, len(tfrecord_paths))

    d = d.apply(
      tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset,
        sloppy=is_training,
        cycle_length=cycle_length))
    d = d.shuffle(buffer_size=1024)
  else:
    d = tf.data.TFRecordDataset(tfrecord_paths)
    # d = d.repeat()

  d = d.apply(
    tf.data.experimental.map_and_batch(
      parser,
      batch_size=flag_objs.batch_size,
      num_parallel_batches=Constants.NUM_CPU_THREADS,
      drop_remainder=True))

  # Get iterator and its initializer
  tfrecord_iterator = d.make_initializable_iterator()
  tfrecord_iterator_initializer = tfrecord_iterator.make_initializer(
    d, name='tfrecord_iterator_initializer')

  # Get feture from tfrecord_iterator ( = Dictionary of Tensors)
  features = tfrecord_iterator.get_next()

  return features, tfrecord_iterator, tfrecord_iterator_initializer


"""
shape
"""


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """

  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
      "For the tensor '%s' in scope '%s', the actual rank "
      "'%d' (shape = %s) is not equal to the expected rank '%s'" %
      (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of 'tensor'. If this is
      specified and the 'tensor' has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


"""
checkpoint
"""


def handle_init_checkpoint(ck_dir, ck_specifier):
  ret = None
  if ck_specifier is None:
    pass
  elif ck_specifier == "ls":
    files = []
    for f in os.listdir(ck_dir):
      if os.path.isfile(os.path.join(ck_dir, f)):
        only_name, ext = os.path.splitext(os.path.basename(f))
        if only_name not in files and '-' in only_name:
          files.append(only_name)

    files.sort(key=lambda x: int(x.split('-')[1]))
    print('[List of checkpoints]')
    for f in files:
      print(f)
    sys.exit(0)
  else:
    init_ck_path = os.path.join(ck_dir, ck_specifier)
    if not tf.gfile.Exists("{}.meta".format(init_ck_path)):
      print('[ERROR] Checkpoint', init_ck_path, 'is not exist')
      sys.exit(0)

    ret = init_ck_path

  return ret


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return assignment_map, initialized_variable_names


"""
metrics
"""


def loss_function(real, pred):
  # loss
  # num_labels = pred.shape[-1]
  # log_probs = tf.nn.log_softmax(pred, axis=-1) + 1e-10
  # one_hot_labels = tf.one_hot(real, depth=num_labels, dtype=tf.float32)
  # loss_ = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss_ = loss_object(real, pred)

  # mask
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  mask = tf.cast(mask, dtype=tf.float32)
  loss_ = tf.multiply(loss_, mask)
  return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(label, prediction):
  mask = tf.math.logical_not(tf.math.equal(label, 0))
  mask = tf.cast(mask, dtype=tf.float32)

  corrects = tf.cast(tf.equal(prediction, label), tf.float32)
  corrects = tf.multiply(corrects, mask)
  return tf.reduce_sum(corrects) / tf.reduce_sum(mask) * 100


"""
optimizer
"""


def train_optimizer(
  loss,
  init_learning_rate,
  num_train_steps,
  num_warmup_steps,
  global_step,
  use_lamb=False
):
  # Implements linear decay of the learning rate.
  learning_rate = tf.constant(value=init_learning_rate, shape=[], dtype=tf.float32)
  learning_rate = tf.train.polynomial_decay(
    learning_rate,
    global_step,
    num_train_steps,
    end_learning_rate=1e-6,
    power=1.0,
    cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int64)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int64)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_learning_rate * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
      (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  if use_lamb:
    tf.logging.info("using lamb")
    optimizer = LAMBOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  else:
    tf.logging.info("using adamw")
    optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  train_op = optimizer.apply_gradients(
    zip(grads, tvars), global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, neither `AdamWeightDecayOptimizer` nor `LAMBOptimizer` do this.
  # But if you use a different optimizer, you should probably take this line
  # out.
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
        name=param_name + "/adam_m",
        shape=param.shape.as_list(),
        dtype=tf.float32,
        trainable=False,
        initializer=tf.zeros_initializer())
      v = tf.get_variable(
        name=param_name + "/adam_v",
        shape=param.shape.as_list(),
        dtype=tf.float32,
        trainable=False,
        initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
        tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
        tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
        [param.assign(next_param),
         m.assign(next_m),
         v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  @staticmethod
  def _get_variable_name(param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


class LAMBOptimizer(tf.train.Optimizer):
  """LAMB (Layer-wise Adaptive Moments optimizer for Batch training)."""
  # A new optimizer that includes correct L2 weight decay, adaptive
  # element-wise updating, and layer-wise justification. The LAMB optimizer
  # was proposed by Yang You, Jing Li, Jonathan Hseu, Xiaodan Song,
  # James Demmel, and Cho-Jui Hsieh in a paper titled as Reducing BERT
  # Pre-Training Time from 3 Days to 76 Minutes (arxiv.org/abs/1904.00962)

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               name="LAMBOptimizer"):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = linalg_ops.norm(param, ord=2)
        g_norm = linalg_ops.norm(update, ord=2)
        ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
            math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      update_with_lr = ratio * self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name