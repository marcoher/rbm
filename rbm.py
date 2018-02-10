from __future__ import division, print_function
import tensorflow as tf
import numpy as np

class RBM(object):
  def __init__(self, visible_dim, hidden_dim, visible_units = "binary", seed = None):
    """
    num_visible   : number of visible units
    num_hidden    : number of hidden units
    visible_units : type of units (binary, gaussian, etc)
    seed          : seed for reproducibility
    """
    self.visible_dim = visible_dim
    self.hidden_dim  = hidden_dim
    self.seed = seed

    self.visible_units = visible_units.strip().lower()
    assert self.visible_units in ["binary"],"units are not implemented: %s" % visible_units
    # plan to add other types of units (multinomial, gaussian, etc)

    if self.visible_units in ["binary", "multinomial"]:
      self.visible_dtype = tf.int32
    elif self.visible_units in ["gaussian"]:
      self.visible_dtype = tf.float32

    self._build()
    self.sess = tf.Session()
    self.sess.run(self.init)

  def _build(self):
    if not self.seed:    self.seed = 1234
    tf.set_random_seed(self.seed)

    self.weights = tf.get_variable(
                    name = "weights",
                    shape = [self.visible_dim, self.hidden_dim],
                    dtype = tf.float32,
                    initializer = tf.random_uniform_initializer(
                        minval = -0.1 * tf.sqrt(6. / (self.hidden_dim + self.visible_dim)),
                        maxval =  0.1 * tf.sqrt(6. / (self.hidden_dim + self.visible_dim))
                        )
                )

    self.v_bias = tf.get_variable(
                    name = "visible_bias",
                    shape = [self.visible_dim],
                    dtype = tf.float32,
                    initializer = tf.random_uniform_initializer()
                )

    self.h_bias = tf.get_variable(
                    name = "hidden_bias",
                    shape = [self.hidden_dim],
                    dtype = tf.float32,
                    initializer = tf.random_uniform_initializer()
                )

    self.visible = tf.placeholder(
                    name = "visible_units",
                    shape = [None, self.visible_dim],
                    dtype = self.visible_dtype
                    )

    self.hidden = tf.placeholder(
                    name = "hidden_units",
                    shape = [None, self.hidden_dim],
                    dtype = tf.int32
                    )

    self.visible_means  = tf.sigmoid(
                            tf.matmul(tf.cast(self.hidden, tf.float32), tf.transpose(self.weights))\
                            + self.v_bias,
                            name = "visible_means"
                            )

    self.visible_sample = tf.where(tf.random_uniform(tf.shape(self.visible_means)) - self.visible_means < 0,
                                   tf.ones_like(self.visible_means),
                                   tf.zeros_like(self.visible_means),
                                   name = "visible_sample"
                              )

    self.hidden_probs  = tf.sigmoid(tf.matmul(tf.cast(self.visible, tf.float32), self.weights)\
                                    + self.h_bias,
                                    name = "hidden_probs"
                                    )

    self.hidden_sample = tf.where(tf.random_uniform(tf.shape(self.hidden_probs)) - self.hidden_probs < 0,
                                  tf.ones_like(self.hidden_probs),
                                  tf.zeros_like(self.hidden_probs),
                                  name = "hidden_sample"
                                  )

    self.init = tf.global_variables_initializer()


  def generate(self, samples = 1, iters = 1, hidden_seed = None):
    """
        Generate samples of visible units
        samples : number of samples to generate
        iters : number of alternating samplings to perform
        hidden_seed : hidden units used to start the sampling
    """
    if hidden_seed:
      h_samples, h_dim = hidden_seed.shape
      assert h_samples==samples,"hidden_seed implies different number of samples: %s" % h_samples
      assert h_dim == self.hidden_dim,"shape of hidden_seed does not match the model: %s" % h_dim
    else:
      shape = [samples, self.hidden_dim]
      hidden_seed = self.sess.run(tf.where(2.0 * tf.random_uniform(shape) < 1,
                                    tf.ones(shape),
                                    tf.zeros(shape))
                                    )

    visible_sample = self.sess.run(self.visible_sample,
                                   feed_dict = {self.hidden: hidden_seed})

    for _ in xrange(iters-1):
      hidden_seed = self.sess.run(self.hidden_sample,
                                  feed_dict = {self.visible : visible_sample})
      visible_sample = self.sess.run(self.visible_sample,
                                     feed_dict = {self.hidden : hidden_seed})

    return visible_sample

  def train(self, data, batch_size = 10, max_epochs = 1000, learning_rate = 0.1, cd_steps = 1):
    """
        Fit the model to a dataset
        data : an array of shape [num_samples, num_hidden]
        max_epochs : maximum number of epochs to use in training
        learning_rate : learning rate to use in training
        cd_steps : number of contrastive divergence steps
    """
    num_samples, visible_dim = data.shape
    assert visible_dim == self.visible_dim,"data dimension does not match the model: %r" % visible_dim

    with tf.Session() as sess:
      sess.run(self.init)
      for epoch in xrange(max_epochs):
        error = self.cd_update(data, learning_rate, cd_steps)
        print( "Epoch %s: reconstruction error = %s" % (epoch, error) )

  def cd_update(self, data, lr, cd_steps):
    """
        Update parameters using contrastive divergence
        data : a rank 2 tensor of shape [:, visible_dim]
        learning_rate : learning rate for gradient ascend
        cd_steps: number of steps for contrastive divergence
    """
    # v0 --> h0 = h | v0
    hidden_probs, hidden_sample = self.sess.run([self.hidden_probs, self.hidden_sample],
                                                feed_dict = {self.visible : data})
    # h0 --> v1 = v | h0
    visible_means, visible_sample = self.sess.run([self.visible_means, self.visible_sample],
                                                  feed_dict = {self.hidden : hidden_sample})
    # reconstruction error: E|v1-v0|^2
    error = self.sess.run(tf.losses.mean_squared_error(data, visible_sample))
    # positive associations : p(v) is given by the data distribution
    # E v_i h_j
    grad_weights = tf.matmul(tf.cast(tf.transpose(data), tf.float32), hidden_probs) / data.shape[0]
    # E v_i
    grad_v_bias  = tf.reduce_mean(tf.cast(data, tf.float32))
    # E h_j
    grad_h_bias  = tf.reduce_mean(hidden_probs)

    # sample h1, v1, ... , h_n-1, v_n
    for _ in xrange(cd_steps-1):
      # v_k --> h_k = h | v_k
      hidden_probs, hidden_sample = self.sess.run([self.hidden_probs, self.hidden_sample],
                                                  feed_dict = {self.visible : visible_sample})
      # h_k --> v_k+1 = v | h_k
      visible_probs, visible_sample = self.sess.run([self.visible_means, self.visible_sample],
                                                    feed_dict = {self.hidden : hidden_sample})

    # negative associations: p(v,h) is given by ( v_n = v | h_n-1, h_n-1)
    # E v_i h_j
    grad_weights -= tf.matmul(tf.transpose(visible_sample), hidden_probs) / data.shape[0]
    # E v_i
    grad_v_bias  -= tf.reduce_mean(visible_means)
    # E h_j
    grad_h_bias  -= tf.reduce_mean(hidden_probs)

    # update
    self.sess.run(tf.assign(self.weights, self.weights + lr * grad_weights))
    self.sess.run(tf.assign(self.v_bias,  self.v_bias  + lr * grad_v_bias))
    self.sess.run(tf.assign(self.h_bias,  self.h_bias  + lr * grad_h_bias))

    return error
