from __future__ import division, print_function
import tensorflow as tf

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

        assert visible_units.strip().lower() in ["binary"],
               "units are not implemented: %s" % visible_units
        #visible_sampler = tf.sigmoid - extensions to come...

        if not seed:    seed = 1234
        tf.set_random_seed(seed)

        self.weights = tf.Variable(
                        name = "weights",
                        shape = [visible_dim, hidden_dim],
                        dtype = tf.float32,
                        initializer = tf.random_uniform_initializer(
                                    minval = -0.1 * np.sqrt(6. / (hidden_dim + visible_dim)),
                                    maxval =  0.1 * np.sqrt(6. / (hidden_dim + visible_dim))
                                    )
                        )

        self.v_bias = tf.Variable(
                        name = "visible_bias",
                        shape = [visible_dim],
                        dtype = tf.float32,
                        initializer = tf.random_uniform_initializer()
                        )

        self.h_bias = tf.Variable(
                        name = "hidden_bias",
                        shape = [hiden_dim],
                        dtype = tf.float32,
                        initializer = tf.random_uniform_initializer()
                        )

    def train(self, data, batch_size = 10, max_epochs = 1000, learning_rate = 0.1, cd_steps = 1):
        """
            Train the model to fit a dataset
            data : a numpy array of shape [num_samples, num_hidden]
            max_epochs : maximum number of epochs to use in training
            learning_rate : learning rate to use in training
            cd_steps : number of contrastive divergence steps
        """
        num_samples, visible_dim = data.shape
        assert visible_dim == self.visible_dim,
               "data dimension does not match the model: %r" % visible_dim

        with tf.Session() as sess:
            sess.global_variables_initializer()
            for epoch in xrange(max_epochs):
                error = self.cd_update(data, learning_rate, cd_steps)
                print("Epoch %s: reconstruction error = %s" % (epoch, error.eval()))

    def generate(self, samples = 1, iters = 1, hidden_seed=None):
        """
            Generate samples of visible units
            samples : number of samples to generate
            iters : number of alternating samplings to perform
            hidden_seed : hidden units used to start the sampling
        """
        if not hidden_seed:
            shape = [samples, self.hidden_dim]
            hidden_seed = tf.where( 2.0 * tf.random_uniform(shape) < 1,
                                    tf.ones(shape),
                                    tf.zeros(shape) )

        visible_sample = self.sample_visible(hidden_seed, probs=False)

        for _ in xrange(iters-1):
            hidden_seed    = self.sample_hidden(visible_sample, probs=False)
            visible_sample = self.sample_visible(hidden_seed, probs=False)

        with tf.Session() as sess:
            return visible_sample.eval()            

    def cd_update(self, data, learning_rate, cd_steps):
        """
            Update parameters using contrastive divergence
            data : a rank 2 tensor of shape [:, visible_dim]
            learning_rate : learning rate for gradient ascend
            cd_steps: number of steps for contrastive divergence
        """
        # v0 --> h0 = h | v0
        hidden_probs, hidden_sample = self.sample_hidden(data)
        # h0 --> v1 = v | h0
        visible_probs, visible_sample = self.sample_visible(hidden_sample)

        # reconstruction error: E|v1-v0|^2
        error = tf.losses.mean_squared_error(data, visible_sample)

        # positive associations : p(v) is given by the data distribution
        # E v_i h_j
        grad_weights = tf.matmul(tf.transpose(data), hidden_probs) / data.shape[0]
        # E v_i
        grad_v_bias  = tf.reduce_mean(data)
        # E h_j
        grad_h_bias = tf.reduce_mean(hidden_probs)

        # sample h1, v1, ... , h_n-1, v_n
        for _ in xrange(cd_steps-1):
            # v_k --> h_k = h | v_k
            hidden_probs, hidden_sample = self.sample_hidden(visible_sample)
            # h_k --> v_k+1 = v | h_k
            visible_probs, visible_sample = self.sample_visible(hidden_sample)

        # negative associations: p(v,h) is given by ( v_n = v | h_n-1, h_n-1)
        # E v_i h_j
        grad_weights -= tf.matmul(tf.transpose(visible_sample), hidden_probs) / data.shape[0]
        # E v_i
        grad_v_bias  -= tf.reduce_mean(visible_probs)
        # E h_j
        grad_h_bias  -= tf.reduce_mean(hidden_probs)

        # update
        self.weights += learning_rate * grad_weights
        self.v_bias  += learning_rate * grad_v_bias
        self.h_bias  += learning_rate * grad_h_bias

        return error

    def sample_hidden(self, visible, probs = True):
        """
            Sample hidden units given the visible units
            visible : a rank 2 tensor of shape [:, visible_dim]
        """
        hidden_probs  = tf.sigmoid(
                            tf.matmul(tf.cast(visible_units, tf.float32), self.weights)\
                            + self.h_bias)

        hidden_sample = tf.where(tf.random_uniform(hidden_probs.shape) - hidden_probs < 0,
                                 tf.ones_like(hidden_probs),
                                 tf.zeros_like(hidden_probs))
        if probs:
            return hidden_probs, hidden_sample
        else:
            return hidden_sample

    def sample_visible(self, hidden, probs = True):
        """
            Sample visible units given the hidden units
            hidden : a rank 2 tensor of shape [:, hidden_dim]
        """
        visible_probs  = tf.sigmoid(
                            tf.matmul(tf.cast(hidden, tf.float32), tf.transpose(self.weights))\
                            + self.v_bias)

        visible_sample = tf.where(tf.random_uniform(visible_probs.shape) - visible_probs < 0,
                                  tf.ones_like(visible_probs),
                                  tf.zeros_like(visible_probs))
        if probs:
            return visible_probs, visible_sample
        else:
            return visible_sample
