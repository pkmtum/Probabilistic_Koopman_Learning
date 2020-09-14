'''
Created on 12 Aug 2020

@author: Tobias Pielok
'''

import tensorflow as tf
import tensorflow.keras as tfk

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras import backend as Kr
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow.python.keras import regularizers

class ExpLayer(tf.keras.layers.Layer):
    
  def __init__(self, init, param_regularizer=None, **kwargs):
    super(ExpLayer, self).__init__(**kwargs)
    self.init = init
    self.param_regularizer = regularizers.get(param_regularizer)

  def vec_to_K(self, params):
    sigma = -(params[0:self.units]**2)
    ceta  = -params[self.units:2*self.units-1]
    
    K = tf.linalg.diag(sigma)
    m_diag = tf.pad(tf.linalg.diag(ceta), [[0, 1], [1, 0]], "CONSTANT")
    return K + m_diag - tf.transpose(m_diag)
    
  def compute_K(self, input):
    self.K =  self.vec_to_K(self.kernel) 
    
  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    
    self.units = (last_dim - 1)

    self.kernel = self.add_weight("kernel",
                                  shape=[int(2*self.units -1)],
                                  initializer=self.init,
                                  regularizer=self.param_regularizer)

  def call(self, input):
    kernel = tf.linalg.expm(input[...,-1] * self.K)
    return tf.matmul(input[...,:self.units], kernel)

class ExpVariational(tf.keras.layers.Layer):
  """Adaption of Dense layer with random `kernel`.


  This layer fits the "weights posterior" according to the following generative
  process:

  ```none
  K ~ Prior()
  M = matmul(X, expm(K))
  Y ~ Likelihood(M)
  ```

  """

  def __init__(self,
               make_posterior_fn,
               make_prior_fn,
               num_train,
               kl_weight=None,
               kl_use_exact=False,
               activation=None,
               activity_regularizer=None,
               **kwargs):
    """Creates the `DenseVariational` layer.

    Arguments:
      make_posterior_fn: Python callable taking `tf.size(kernel)`,
        `tf.size(bias)`, `dtype` and returns another callable which takes an
        input and produces a `tfd.Distribution` instance.
      make_prior_fn: Python callable taking `tf.size(kernel)`, `tf.size(bias)`,
        `dtype` and returns another callable which takes an input and produces a
        `tfd.Distribution` instance.
      kl_weight: Amount by which to scale the KL divergence loss between prior
        and posterior.
      kl_use_exact: Python `bool` indicating that the analytical KL divergence
        should be used rather than a Monte Carlo approximation.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      **kwargs: Extra arguments forwarded to `tf.keras.layers.Layer`.
    """
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)
    super(ExpVariational, self).__init__(
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
        **kwargs)

    self._make_posterior_fn = make_posterior_fn
    self._make_prior_fn = make_prior_fn
    self._kl_divergence_fn = _make_kl_divergence_penalty(
        kl_use_exact, weight=kl_weight)

    self.activation = tf.keras.activations.get(activation)
    self.supports_masking = False
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
    self.num_train = num_train

  def vec_to_K(self, params):
    sigma = -(tf.math.softplus(params[0:self.units])**2)
    ceta  = -params[self.units:2*self.units-1]
    
    K = tf.linalg.diag(sigma)
    m_diag = tf.pad(tf.linalg.diag(ceta), [[0, 1], [1, 0]], "CONSTANT")
    return K + m_diag - tf.transpose(m_diag)
    
  def build(self, input_shape):
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `DenseVariational` '
                       'should be defined. Found `None`.')
    
    self.units = (last_dim - 1)

    self.input_spec = tf.keras.layers.InputSpec(
        min_ndim=2, axes={-1: last_dim})

    self._posterior = self._make_posterior_fn(
       2*self.units -1,
        dtype)
    self._prior = self._make_prior_fn(
        2*self.units -1,
        dtype)

    self.built = True


  def compute_K(self, inputs):
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    inputs = tf.cast(inputs, dtype, name='inputs')

    q = self._posterior(inputs)
    r = self._prior(inputs)
    
    self.add_loss(self._kl_divergence_fn(q, r)/ (self.num_train * 1.0))

    w = tf.convert_to_tensor(value=q)
    self.K =  self.vec_to_K(w)   
    
  def call(self, inputs):    
    kernel = tf.linalg.expm(inputs[...,-1] * self.K)
    outputs = tf.matmul(inputs[...,:self.units], kernel)

    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable
    
    return outputs
    #return tf.concat([outputs, [[self._kl_divergence_fn(q, r)]]],1)#outputs


def _make_kl_divergence_penalty(
    use_exact_kl=False,
    test_points_reduce_axis=(),  # `None` == "all"; () == "none".
    test_points_fn=tf.convert_to_tensor,
    weight=None):
  """Creates a callable computing `KL[a,b]` from `a`, a `tfd.Distribution`."""

  if use_exact_kl:
    kl_divergence_fn = kullback_leibler.kl_divergence
  else:
    def kl_divergence_fn(distribution_a, distribution_b):
      z = test_points_fn(distribution_a)
      return tf.reduce_mean(
          distribution_a.log_prob(z) - distribution_b.log_prob(z),
          axis=test_points_reduce_axis)

  # Closure over: kl_divergence_fn, weight.
  def _fn(distribution_a, distribution_b):
    """Closure that computes KLDiv as a function of `a` as in `KL[a, b]`."""
    with tf.name_scope('kldivergence_loss'):
      kl = kl_divergence_fn(distribution_a, distribution_b)
      if weight is not None:
        kl = tf.cast(weight, dtype=kl.dtype) * kl
      # Losses appended with the model.add_loss and are expected to be a single
      # scalar, unlike model.loss, which is expected to be the loss per sample.
      # Therefore, we reduce over all dimensions, regardless of the shape.
      # We take the sum because (apparently) Keras will add this to the *post*
      # `reduce_sum` (total) loss.
      # TODO(b/126259176): Add end-to-end Keras/TFP test to ensure the API's
      # align, particularly wrt how losses are aggregated (across batch
      # members).
      return tf.reduce_sum(kl, name='batch_total_kl_divergence')

  return _fn

class ExpDistributed(tfk.layers.Wrapper):
  """Adaption from TimeDistributed Layer
  
  Call arguments:
    inputs: Input tensor.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the
      wrapped layer (only if the layer supports this argument).
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked. This argument is passed to the
      wrapped layer (only if the layer supports this argument).

  Raises:
    ValueError: If not initialized with a `tf.keras.layers.Layer` instance.
  """

  def __init__(self, units, layer, **kwargs):
    if not isinstance(layer, Layer):
      raise ValueError(
          'Please initialize `ExpDistributed` layer with a '
          '`tf.keras.layers.Layer` instance. You passed: {input}'.format(
              input=layer))
    super(ExpDistributed, self).__init__(layer, **kwargs)
    self.supports_masking = True
    self._supports_ragged_inputs = True

    self.units = units
    
    # It is safe to use the fast, reshape-based approach with all of our
    # built-in Layers.
    self._always_use_reshape = (
        layer_utils.is_builtin_layer(layer) and
        not getattr(layer, 'stateful', False))

  def _get_shape_tuple(self, init_tuple, tensor, start_idx, int_shape=None):
    """Finds non-specific dimensions in the static shapes.

    The static shapes are replaced with the corresponding dynamic shapes of the
    tensor.

    Arguments:
      init_tuple: a tuple, the first part of the output shape
      tensor: the tensor from which to get the (static and dynamic) shapes
        as the last part of the output shape
      start_idx: int, which indicate the first dimension to take from
        the static shape of the tensor
      int_shape: an alternative static shape to take as the last part
        of the output shape

    Returns:
      The new int_shape with the first part from init_tuple
      and the last part from either `int_shape` (if provided)
      or `tensor.shape`, where every `None` is replaced by
      the corresponding dimension from `tf.shape(tensor)`.
    """
    # replace all None in int_shape by Kr.shape
    if int_shape is None:
      int_shape = Kr.int_shape(tensor)[start_idx:]
    if not any(not s for s in int_shape):
      return init_tuple + tuple(int_shape)
    shape = Kr.shape(tensor)
    int_shape = list(int_shape)
    for i, s in enumerate(int_shape):
      if not s:
        int_shape[i] = shape[start_idx + i]
    return init_tuple + tuple(int_shape)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if len(input_shape) < 3:
      raise ValueError(
          '`ExpDistributed` Layer should be passed an `input_shape ` '
          'with at least 3 dimensions, received: ' + str(input_shape))
    # Don't enforce the batch or time dimension.
    self.input_spec = InputSpec(shape=[None, None] + input_shape[2:])
    child_input_shape = [input_shape[0]] + input_shape[2:]
    super(ExpDistributed, self).build(tuple(child_input_shape))
    
    self.built = True
    

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    child_input_shape = tensor_shape.TensorShape([input_shape[0]] +
                                                 input_shape[2:])
    child_output_shape = self.layer.compute_output_shape(child_input_shape)
    if not isinstance(child_output_shape, tensor_shape.TensorShape):
      child_output_shape = tensor_shape.TensorShape(child_output_shape)
    child_output_shape = child_output_shape.as_list()
    timesteps = input_shape[1]
    return tensor_shape.TensorShape([child_output_shape[0], timesteps] +
                                    child_output_shape[1:])

  def call(self, inputs, training=None, mask=None):
    kwargs = {}
    if generic_utils.has_arg(self.layer.call, 'training'):
      kwargs['training'] = training

    input_shape = Kr.int_shape(inputs)
    
    inputs, row_lengths = Kr.convert_inputs_if_ragged(inputs)
    is_ragged_input = row_lengths is not None
    
    self.layer.compute_K(inputs)
    
    # batch size matters, use rnn-based implementation
    def step(x, _):
        output = self.layer(x, **kwargs)
        return output, []
    
    _, outputs, _ = Kr.rnn(
          step,
          inputs,
          initial_states=[],
          input_length=row_lengths[0] if is_ragged_input else input_shape[1],
          mask=mask,
          unroll=False)

    y = Kr.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)

    return y

  def compute_mask(self, inputs, mask=None):
    """Computes an output mask tensor for Embedding layer.

    This is based on the inputs, mask, and the inner layer.
    If batch size is specified:
    Simply return the input `mask`. (An rnn-based implementation with
    more than one rnn inputs is required but not supported in tf.keras yet.)
    Otherwise we call `compute_mask` of the inner layer at each time step.
    If the output mask at each time step is not `None`:
    (E.g., inner layer is Masking or RNN)
    Concatenate all of them and return the concatenation.
    If the output mask at each time step is `None` and the input mask is not
    `None`:(E.g., inner layer is Dense)
    Reduce the input_mask to 2 dimensions and return it.
    Otherwise (both the output mask and the input mask are `None`):
    (E.g., `mask` is not used at all)
    Return `None`.

    Arguments:
      inputs: Tensor with shape [batch size, timesteps, ...] indicating the
        input to ExpDistributed. If static shape information is available for
        "batch size", `mask` is returned unmodified.
      mask: Either None (indicating no masking) or a Tensor indicating the
        input mask for ExpDistributed. The shape can be static or dynamic.

    Returns:
      Either None (no masking), or a [batch size, timesteps, ...] Tensor with
      an output mask for the ExpDistributed layer with the shape beyond the
      second dimension being the value of the input mask shape(if the computed
      output mask is none), an output mask with the shape beyond the first
      dimension being the value of the mask shape(if mask is not None) or
      output mask with the shape beyond the first dimension being the
      value of the computed output shape.

    """
    # cases need to call the layer.compute_mask when input_mask is None:
    # Masking layer and Embedding layer with mask_zero
    input_shape = Kr.int_shape(inputs)
    if input_shape[0] and not self._always_use_reshape or isinstance(
        inputs, ragged_tensor.RaggedTensor):
      # batch size matters, we currently do not handle mask explicitly, or if
      # the layer always uses reshape approach, or the input is a ragged tensor.
      return mask
    inner_mask = mask
    if inner_mask is not None:
      inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
      inner_mask = Kr.reshape(inner_mask, inner_mask_shape)
    inner_input_shape = self._get_shape_tuple((-1,), inputs, 2)
    inner_inputs = array_ops.reshape(inputs, inner_input_shape)
    output_mask = self.layer.compute_mask(inner_inputs, inner_mask)
    if output_mask is None:
      if mask is None:
        return None
      # input_mask is not None, and output_mask is None:
      # we should return a not-None mask
      output_mask = mask
      for _ in range(2, len(Kr.int_shape(mask))):
        output_mask = Kr.any(output_mask, axis=-1)
    else:
      # output_mask is not None. We need to reshape it
      input_length = input_shape[1]
      if not input_length:
        input_length = Kr.shape(inputs)[1]
      output_mask_int_shape = Kr.int_shape(output_mask)
      if output_mask_int_shape is None:
        # if the output_mask does not have a static shape,
        # its shape must be the same as mask's
        if mask is not None:
          output_mask_int_shape = Kr.int_shape(mask)
        else:
          output_mask_int_shape = Kr.compute_output_shape(input_shape)[:-1]
      output_mask_shape = self._get_shape_tuple(
          (-1, input_length), output_mask, 1, output_mask_int_shape[1:])
      output_mask = Kr.reshape(output_mask, output_mask_shape)
    return output_mask
