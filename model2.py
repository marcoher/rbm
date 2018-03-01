from __future__ import division, print_function
import tensorflow as tf
import numpy as np

class RBM2(object):
  """
  Boltzmann Machine with two identical visible layers
  """
  def __init__(self, v_dim, h_dim, v_type="binary", seed=0, name="rbm2"):
    """
    v_dim  : number of visible units
    h_dim  : number of hidden units
    v_type : optional, type of visible units, default is binary
    seed   : optional, seed for reproducibility, default is 0
    name   : optional, name of the model, default is "rbm"
    """
    implemented = ["binary", "gaussian"]
    assert (isinstance(v_dim, int) and v_dim>0),"v_dim must be a positive integer: %s" % v_dim
    assert (isinstance(h_dim, int) and h_dim>0),"h_dim must be a positive integer: %s" % h_dim
    assert v_type.strip().lower() in implemented,"v_type is not implemented: %s" % v_type
    assert isinstance(seed, int),"seed must be an integer: %s" % seed
    assert isinstance(name, str),"name must be a string: %s"   % name

    self.v_dim  = v_dim
    self.h_dim  = h_dim
    self.seed   = seed
    self.v_type = v_type.strip().lower()
    self.name   = name
    self.model  = "./" + name + ".ckpt"
    self.cd_steps = None
    self.trained  = False
    self._build()

  def fit(self, X, X_eval=None, max_epochs=100, update_tol=0.001, limit_updates=3, lr=0.01, cd_steps=1, verbose=10, warm_start=False):
    """
    Fit the model to a dataset
    X          : array-like of shape [:, v_dim, 2] used to fit the model
    X_eval     : optional, array-like of shape [:, v_dim, 2] used for evaluation, default is X
    max_epochs : optional, maximum number of epochs , default is 100
    lr         : optional, learning rate, default is 0.01
    cd_steps   : optional, number of contrastive divergence steps, default is 1
    verbose    : optional, print training summary every verbose epochs, default is 10
    """
    X = np.array(X)
    if X_eval is None: X_eval = X
    assert (len(X.shape)==3 and X.shape[2]==2 and
           X.shape[1]==self.v_dim),("shape of X does not match the model: %s" % X.shape)
    assert (isinstance(max_epochs, int) and max_epochs>0),"max_epochs must be a positive integer: %s" % max_epochs
    assert (isinstance(lr, float) and lr>0),"lr must be a positive floating point number: %s" % lr
    assert (isinstance(cd_steps, int) and cd_steps>0),"cd_steps must a positive ingeger: %s" % cd_steps
    assert (isinstance(verbose, int) and verbose>0),"verbose must be a positive integer: %s" % verbose

    if not self.cd_steps:
      self.cd_steps = cd_steps
      self._training_ops(cd_steps)
    else:
      assert (self.cd_steps==cd_steps),"cd_steps does not match previous value: %d (expected %d)" % (cd_steps, self.cd_steps)

    train_dict = {self.v1: X[:,:,0],
                  self.v2: X[:,:,1],
                  self.lr: lr,
                  self.N: X.shape[0] + 0.0}
    ct = 0
    with tf.Session(graph=self.graph) as sess:
      if self.trained:
        self.saver.restore(sess, self.model)
      else:
        sess.run(self.init)
      for epoch in xrange(max_epochs+1):
        upd8,_ = sess.run([self.max_update, self.update], feed_dict=train_dict)
        if upd8<update_tol:
          ct += 1
        else:
          ct = 0
        if ct>limit_updates:
          print("Update tolerance has been below threshold %s for more than %d iterations. Tranining stops now." % (update_tol, limit_updates))
          break
        if epoch % verbose == 0:
          f = sess.run(self._free_energy, feed_dict={self.v1: X_eval[:,:,0], self.v2: X_eval[:,:,1]})
          print("Epoch %d/%d: free energy = %s" % (epoch, max_epochs, np.mean(f)))


      saved_path = self.saver.save(sess, self.model)

    self.trained = True
    print("\nTraining complete. Model saved in: %s\n" % saved_path)
    print("Number of samples: %d, learning rate: %.4f, cd steps: %d\n" % (X.shape[0], lr, cd_steps))


  def generate(self, v, samples=1, iters=1):
    """
    Generate random samples of the visible units
    v       : array-like of shape [v_dim], to clamp the generations
    samples : optional, number of samples to generate, default is 1
    iters   : optional, number of hidden-visible alternations to use when sampling, default is 1

    output  : array-like of shape [samples, v_dim], samples of the second visible layer
    """

    assert (isinstance(samples, int) and samples>0),"samples must be a positive integer: %s" % samples
    assert (isinstance(iters, int) and iters>0),"iters must be a positive integer: %s" % iters

    h = np.random.random_integers(0, 1, (samples, self.h_dim))
    v1 = np.repeat(v.reshape((1,-1)), samples, axis=0)

    with tf.Session(graph=self.graph) as sess:
      if self.trained:
        self.saver.restore(sess, self.model)
      else:
        sess.run(self.init)
      for _ in xrange(iters):
          v2 = sess.run(self.v_given_hv, feed_dict={self.v1: v1, self.h: h})
          h = sess.run(self.h_given_vv, feed_dict={self.v1: v1, self.v2: v2})

    return v2


  def free_energy(self, v1, v2):
    """
    Free energy of visible layers

    v1 : array-like of shape [samples, v_dim]
    v2 : array-like of shape [samples, v_dim]

    output : array-like of shape [samples]
    """
    with tf.Session(graph=self.graph) as sess:
      if self.trained:
        self.saver.restore(sess, self.model)
      else:
        sess.run(self.init)
      F = sess.run(self._free_energy, feed_dict={self.v1: v1, self.v2: v2})

    return F


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
      self.v_given_hv, _ = self._conditional_v_sample(self.v1, self.h)
      self.h_given_vv, _ = self._conditional_h_sample(self.v1, self.v2)
      self._energy_nodes()
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver()


  def _units(self):
    """
    Set the data type of the visible units
    """
    if self.v_type in ["binary"]:
      self.v_dtype = tf.uint8
    elif self.v_type in ["gaussian"]:
      self.v_dtype = tf.float32


  def _variables(self):
    """
    Variables of the model
    """
    with tf.name_scope("variables"):
      self.W0 = tf.get_variable(
               name="weights_hv",
               shape=[self.v_dim, self.h_dim],
               dtype=tf.float32,
               initializer=tf.truncated_normal_initializer(stddev=0.01))

      self.WI = tf.get_variable(
               name="weights_vv",
               shape=[self.v_dim, self.v_dim],
               dtype=tf.float32,
               initializer=tf.zeros_initializer())

      self.vbias = tf.get_variable(
                  name="visible_bias",
                  shape=[self.v_dim],
                  dtype=tf.float32,
                  initializer=tf.zeros_initializer())

      self.hbias = tf.get_variable(
                  name="hidden_bias",
                  shape=[self.h_dim],
                  dtype=tf.float32,
                  initializer=tf.zeros_initializer())


  def _random_units(self):
    """
    Placeholders for the states of the visible and hidden units
    """
    with tf.name_scope("random_units"):
      self.v1 = tf.placeholder(name="visible_layer_1",
                               shape=[None, self.v_dim],
                               dtype=self.v_dtype)

      self.v2 = tf.placeholder(name="visible_layer_2",
                               shape=[None, self.v_dim],
                               dtype=self.v_dtype)

      self.h = tf.placeholder(name="hidden_layer",
                              shape=[None, self.h_dim],
                              dtype = tf.uint8)


  def _conditional_v_sample(self, v, h):
    """
    Add ops to the main graph to sample values of the visible units given v,h

    v : tensor of shape [:, v_dim], already in the graph
    h : tensor of shape [:, h_dim], already in the graph

    output : tensor of shape [:, v_dim]
    """
    with tf.name_scope("v_given_h"):
      v_acts = ( tf.matmul(tf.cast(h, tf.float32), tf.transpose(self.W0))
               + tf.matmul(tf.cast(v, tf.float32), self.WI)
               + self.vbias )

      if self.v_type=="binary":
        v_means = tf.sigmoid(v_acts, name="v_means")

        v_sample = tf.where(
                  tf.random_uniform(tf.shape(v_means)) - v_means<0.0,
                  tf.ones_like(v_means, dtype=self.v_dtype),
                  tf.zeros_like(v_means, dtype=self.v_dtype),
                  name = "v_sample")

      elif self.v_type=="gaussian":
        v_means = tf.identity(v_acts, name="v_means")

        v_sample = tf.add(v_means,
                          tf.truncated_normal(tf.shape(v_means)),
                          name="v_sample")

    return v_sample, v_means


  def _conditional_h_sample(self, v1, v2):
    """
    Add ops to the main graph to sample values for the hidden units given v

    v1 : tensor of shape [:, v_dim] in the graph
    v2 : tensor of shape [:, v_dim] in the graph

    output : tensor of shape [:, h_dim]
    """
    with tf.name_scope("hidden_given_visible"):
      h_probs = tf.sigmoid(tf.matmul(tf.cast(v1 + v2, tf.float32),
                                     self.W0)
                           + self.hbias,
                           name="hidden_probs")

      h_sample = tf.where(
                tf.random_uniform(tf.shape(h_probs)) - h_probs<0,
                tf.ones_like(h_probs, dtype=tf.uint8),
                tf.zeros_like(h_probs, dtype=tf.uint8),
                name="hidden_sample")

    return h_sample, h_probs


  def _training_ops(self, k):
    """
    Add ops to the main graph to perform k-step cylcing updates:
    each cycle is: (v1,v2) -> h, (v2,h) -> v1, (h,v1) -> v2, (v1,v2) -> h

    k : number of cycles
    """
    with self.graph.as_default():
      self.lr = tf.placeholder(name="learning_rate", shape=(), dtype=tf.float32)
      self.N  = tf.placeholder(name="num_samples", shape=(), dtype=tf.float32)

      v0_1, v0_2 = self.v1, self.v2
      h0, Eh0 = self._conditional_h_sample(v0_1, v0_2)

      v1, Ev1 = self._conditional_v_sample(v0_2, h0)
      v2, Ev2 = self._conditional_v_sample(v1, h0)
      h1, Eh1 = self._conditional_h_sample(v1, v2)

      for _ in xrange(k-1):
        v0_1, v0_2, h0 = v1, v2, h1
        v1, Ev1 = self._conditional_v_sample(v0_2, h0)
        v2, Ev2 = self._conditional_v_sample(v1, h0)
        h1, Eh1 = self._conditional_h_sample(v1, v2)

      grad_W0 = ( tf.matmul(tf.transpose(tf.cast(self.v1 + self.v2, tf.float32)),
                            Eh0) / self.N
                - tf.matmul(tf.transpose(tf.cast(v1 + v2, tf.float32)),
                            Eh1) / self.N )

      max_update = tf.norm(grad_W0, ord=np.inf)

      grad_WI = ( tf.matmul(tf.transpose(tf.cast(self.v2, tf.float32)),
                            tf.cast(self.v1, tf.float32)) / self.N
                - tf.matmul(tf.transpose(Ev2),
                            tf.cast(v1, tf.float32)) / self.N )
      max_update = tf.maximum(max_update, tf.norm(grad_WI, ord=np.inf))

      grad_vbias = tf.reduce_mean(
                  tf.cast(self.v1 + self.v2, tf.float32) - (Ev1 + Ev2),
                  axis=0)
      max_update = tf.maximum(max_update, tf.norm(grad_vbias, ord=np.inf))

      grad_hbias = tf.reduce_mean(Eh0 - Eh1, axis=0)
      max_update = tf.maximum(max_update, tf.norm(grad_hbias, ord=np.inf))

      W0_update  = self.W0.assign_add(self.lr * grad_W0)

      WI_update  = self.WI.assign_add(
                  (self.lr/2.0) * (grad_WI + tf.transpose(grad_WI)))

      vbias_update = self.vbias.assign_add(self.lr * grad_vbias)
      hbias_update = self.hbias.assign_add(self.lr * grad_hbias)

      self.update = [W0_update, WI_update, vbias_update, hbias_update]
      self.max_update = self.lr * max_update


  def _energy_nodes(self):
    with tf.name_scope("free_energy_ops"):
      vs = tf.cast(self.v1 + self.v2, tf.float32)
      bias_part = tf.tensordot(vs, self.vbias, axes=[[-1],[0]])
      int_part = tf.reduce_sum(tf.matmul(tf.cast(self.v1, tf.float32), self.WI)
                               * tf.cast(self.v2, tf.float32),
                               axis=-1)
      h_part = tf.reduce_sum(tf.nn.softplus(tf.matmul(vs, self.W0) + self.hbias),
                             axis=-1)

      self._free_energy = - bias_part - int_part - h_part
