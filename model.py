from __future__ import division, print_function
import tensorflow as tf
import numpy as np

implemented = ["binary", "gaussian"] # TODO: implement binomial, etc

class rbm(object):
  """
  Restricted Boltzmann Machine
  """
  def __init__(self, visible_dim, hidden_dim, visible_type = "binary", seed = 1234, name = "rbm"):
    """
    visible_dim   : number of visible units
    hidden_dim    : number of hidden units
    visible_type  : optional, type of visible units, default is binary
    seed          : optional, seed for reproducibility, default is 1234
    name          : optional, name of the model, default is rbm
    """

    assert (isinstance(visible_dim,int) and visible_dim>0),"visible_dim must be a positive integer: %s" % visible_dim
    assert (isinstance(hidden_dim, int) and hidden_dim>0 ),"hidden_dim must be a positive integer: %s" % hidden_dim
    assert visible_type.strip().lower() in implemented,"visible_type is not implemented: %s (expected in %s)" % (visible_type, implemented)
    assert isinstance(seed, int),"seed must be an integer: %s" % seed
    assert isinstance(name, str),"name must be a string: %s" % name

    self.visible_dim  = visible_dim
    self.hidden_dim   = hidden_dim
    self.seed         = seed
    self.visible_type = visible_type.strip().lower()
    self.name         = name
    self.cd_steps     = None
    self.trained      = False
    self.model_files  = "./tmp/" + name + ".ckpt"
    self._build()

  def fit(self, data, max_epochs = 1000, learning_rate = 0.1, cd_steps = 1, verbose = 100):
    """
    Fit the model to a dataset
    data          : a numpy array of shape [num_samples, visible_dim]
    max_epochs    : optional, maximum number of epochs to use in training, default is 1000
    learning_rate : optional, learning rate to use in training, default is 0.1
    cd_steps      : optional, number of contrastive divergence steps, default is 1
    verbosity     : optional, print reconstruction error every verbose epochs, default is 100
    """

    assert isinstance(data, np.ndarray),"data must be a numpy array: %s" % data
    assert (len(data.shape)==2 and data.shape[1]==self.visible_dim),"shape of data does not match the model: %s" % data.shape
    assert (isinstance(max_epochs, int) and max_epochs>0),"max_epochs must be a positive integer: %s" % max_epochs
    assert (isinstance(learning_rate, float) and learning_rate>0),"learning_rate must be a positive floating point number: %s" % learning_rate
    assert (isinstance(cd_steps, int) and cd_steps>0),"cd_steps must a positive ingeger: %s" % cd_steps
    assert (isinstance(verbose, int) and verbose>0),"verbose must be a positive integer: %s" % verbose

    self._training_ops(cd_steps)

    train_dict = {self.visible : data, self.lr : learning_rate, self.N : data.shape[0]}

    with tf.Session(graph=self.graph) as sess:
      if self.trained:
        self.saver.restore(sess, self.model_files)
      else:
        sess.run(self.init)

      for epoch in xrange(max_epochs):
        sess.run(self.update, feed_dict = train_dict)
        if epoch % verbose == 0:
            error, free_energy = sess.run([self.reconstruction_error, self.avg_free_energy],
                                          feed_dict = {self.visible : data})
            print("Epoch %d/%d:\nreconstruction error = %.6f, free energy = %8.4f" % (epoch, max_epochs, error, free_energy))
      saved_path = self.saver.save(sess, self.model_files)

    self.trained = True
    print("\nTraining complete. Model saved in: %s\n" % saved_path)
    print("Number of samples: %d, learning rate: %.6f, cd steps: %d\n" % (data.shape[0], learning_rate, cd_steps))



  def generate(self, num_samples = 1, iters = 1):
    """
    Generate random samples of the visible units
    num_samples : optional, number of samples to generate, default is 1
    iters       : optional, number of hidden-visible alternations to use when sampling, default is 1
    """

    assert (isinstance(num_samples, int) and num_samples>0),"num_samples must be a positive integer: %s" % num_samples
    assert (isinstance(iters, int) and iters>0),"iters must be a positive integer: %s" % iters

    hidden_seed = np.random.random_integers(0, 1, (num_samples, self.hidden_dim))

    with tf.Session(graph=self.graph) as sess:
      if self.trained:
        self.saver.restore(sess, self.model_files)
      else:
        sess.run(self.init)
      for _ in xrange(iters):
          visible = sess.run(self.visible_sample, feed_dict={self.hidden : hidden_seed})
          hidden_seed = sess.run(self.hidden_sample, feed_dict={self.visible : visible})

    return visible

  def free_energy(self, v, average=True):
    """
    Free energy of a visible vector
    v : an array of shape [:, visible_dim]
    """
    with tf.Session(graph=self.graph) as sess:
      if self.trained:
        self.saver.restore(sess, self.model_files)
      else:
        sess.run(self.init)
      if average:
        return sess.run(self.avg_free_energy, feed_dict={self.visible : v})
      else:
        return sess.run(self.free_energy_op, feed_dict={self.visible : v})

  def _build(self):
    """
    Build the main graph
    """
    self.graph = tf.Graph()
    with self.graph.as_default():
      tf.set_random_seed(self.seed)
      self._units()
      self._variables()
      self._random_units()
      self.visible_sample, self.visible_means = self._conditional_v_sample(self.hidden)
      self.hidden_sample,  self.hidden_probs, self.hidden_acts  = self._conditional_h_sample(self.visible)
      self._free_energy_op()
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver()

  def _units(self):
    """
    Set the data type of the visible units
    """
    #this should be extended to account for other types of visible units
    if self.visible_type == "binary":
      self.visible_dtype = tf.uint8
    elif self.visible_type == "gaussian":
      self.visible_dtype = tf.float32

  def _variables(self):
    """
    Variables of the model
    """
    with tf.variable_scope("variables") as scope:
      self.weights = tf.get_variable(
                  name = "weights",
                  shape = [self.visible_dim, self.hidden_dim],
                  dtype = tf.float32,
                  initializer = tf.truncated_normal_initializer(stddev=0.01))

      self.v_bias = tf.get_variable(
                  name = "visible_bias",
                  shape = [self.visible_dim],
                  dtype = tf.float32,
                  initializer = tf.zeros_initializer())

      self.h_bias = tf.get_variable(
                  name = "hidden_bias",
                  shape = [self.hidden_dim],
                  dtype = tf.float32,
                  initializer = tf.zeros_initializer())

  def _random_units(self):
    """
    Placeholders for the states of the visible and hidden units
    """
    with tf.name_scope("random_units"):
      self.visible = tf.placeholder(
                    name = "visible_units",
                    shape = [None, self.visible_dim],
                    dtype = self.visible_dtype
                    )

      self.hidden = tf.placeholder(
                    name = "hidden_units",
                    shape = [None, self.hidden_dim],
                    dtype = tf.uint8
                    )

  def _conditional_v_sample(self, h):
    """
    Add ops to the main graph to sample values of the visible units given h
    h : a tensor of shape [:, hidden_dim], must already be in the graph
    """
    #this should be extended to account for other types of visible units
    with tf.name_scope("visible_given_hidden"):
      v_acts = tf.add(tf.matmul(tf.cast(h, tf.float32), tf.transpose(self.weights)),
                      self.v_bias,
                      name="visible_activations")

      if self.visible_type=="binary":
        v_means  = tf.sigmoid(v_acts, name = "visible_means")

        v_sample = tf.where(tf.random_uniform(tf.shape(v_means)) - v_means < 0,
                            tf.ones_like(v_means, dtype = self.visible_dtype),
                            tf.zeros_like(v_means, dtype = self.visible_dtype),
                            name = "visible_sample")

      elif self.visible_type=="gaussian":
        v_means  = tf.identity(v_acts, name = "visible_means")

        v_sample = tf.add(v_means, tf.truncated_normal(), name = "visible_sample")

    return v_sample, v_means

  def _conditional_h_sample(self, v):
    """
    Add ops to the main graph to sample values of the hidden units given v
    v : a tensor of shape [:, visible_dim], must already be in the graph
    """
    with tf.name_scope("hidden_given_visible"):
      h_acts   = tf.add(tf.matmul(tf.cast(v, tf.float32), self.weights), self.h_bias, name="hidden_activations")
      h_probs  = tf.sigmoid(h_acts, name = "hidden_probs")
      h_sample = tf.where(tf.random_uniform(tf.shape(h_probs)) - h_probs < 0,
                          tf.ones_like(h_probs, dtype = tf.uint8),
                          tf.zeros_like(h_probs, dtype = tf.uint8),
                          name = "hidden_sample")

    return h_sample, h_probs, h_acts

  def _training_ops(self, k):
    """
    Add ops to the main graph to perform k-step Contrasting Divergence updates
    k : number of steps in contrastive divergence (cannot be modified)
    """
    #TODO : implement this so that multiple calls can use different k
    if self.cd_steps:   return None

    with self.graph.as_default():
      self.lr = tf.placeholder(name = "learning_rate", shape = (), dtype = tf.float32)
      self.N  = tf.placeholder(name = "num_samples", shape = (), dtype = tf.int32)

      v0 = self.visible
      h0 = self.hidden_sample

      v1, v1_means = self._conditional_v_sample(h0)
      h1, h1_probs,_ = self._conditional_h_sample(v1)

      self.reconstruction_error = tf.losses.mean_squared_error(v0, v1)

      for _ in xrange(k-1):
        v0, h0 = v1, h1
        v1, v1_means = self._conditional_v_sample(h0)
        h1, h1_probs,_ = self._conditional_h_sample(v1)

      grad_w = ( tf.matmul(tf.transpose(tf.cast(self.visible, tf.float32)), self.hidden_probs)\
               - tf.matmul(tf.transpose(tf.cast(v1, tf.float32)), h1_probs)    ) / tf.cast(self.N, tf.float32)

      grad_bv = tf.reduce_mean(tf.cast(self.visible, tf.float32) - v1_means, axis=0)

      grad_bh = tf.reduce_mean(tf.cast(self.hidden_probs,  tf.float32) - h1_probs, axis=0)

      w_update  = self.weights.assign_add(self.lr * grad_w)
      bv_update = self.v_bias.assign_add( self.lr * grad_bv)
      bw_update = self.h_bias.assign_add( self.lr * grad_bh)

      self.update = [w_update, bv_update, bw_update]

  def _free_energy_op(self):
    """
    Add ops to compute the free energy of a visible vector
    """
    with tf.name_scope("free_energy"):
      self.free_energy_op = - tf.add(tf.tensordot(tf.cast(self.visible, tf.float32), self.v_bias, axes=[[-1],[0]]),
                                     tf.reduce_sum(tf.log(1.0 + tf.exp(self.hidden_acts)), axis=-1),
                                     name = "free_energy")

      self.avg_free_energy = tf.reduce_mean(self.free_energy_op)
