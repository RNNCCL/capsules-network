import tensorflow as tf
import numpy as np


class CapsuleNetwork(object):
    def __init__(self, n_inputs, routing_iters=3, optimizer=tf.train.AdamOptimizer(learning_rate=0.001)):
        # self.graph = tf.Graph()
        # with self.graph.as_default():

        self.n_inputs = n_inputs
        self.n_routing_iters = routing_iters

        self.X_in = tf.placeholder(tf.float32, (self.n_inputs, 784))
        self.X = tf.reshape(self.X_in, (self.n_inputs, 28, 28, 1))
        self.Y_in = tf.placeholder(tf.float32, (self.n_inputs, 10))
        self.Y = tf.reshape(self.Y_in, (self.n_inputs, 1, 10))
        self.d_capsules = self._network()

        self.cost = self.cost()
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _network(self):
        with tf.variable_scope('Convolution'):
            convs = tf.contrib.layers.conv2d(self.X, num_outputs=256, kernel_size=9, stride=1, padding='VALID')   # ReLU activation

        with tf.variable_scope('PrimaryCapsules'):
            # 32 is number of capsule blocks
            p_capsules = tf.contrib.layers.conv2d(convs, num_outputs=8 * 32, kernel_size=9, stride=2,
                                                       padding='VALID', activation_fn=None)    # activation replaced by squashing function
            p_capsules = tf.reshape(p_capsules, (self.n_inputs, -1, 1, 8, 1))   # add dim for routing

            # squash
            p_capsules_norm = tf.norm(p_capsules, axis=3, keep_dims=True)
            p_capsules = p_capsules_norm / (1 + p_capsules_norm) * p_capsules / tf.square(p_capsules_norm)

        with tf.variable_scope('Routing'):
            w = tf.get_variable('weight_matrices', shape=(1, 1152, 10, 8, 16), dtype=tf.float32, initializer=tf.random_normal_initializer())

            u_i = tf.matmul(tf.tile(w, [self.n_inputs, 1, 1, 1, 1]), tf.tile(p_capsules, [1, 1, 10, 1, 1]), transpose_a=True)

            # routing logits
            b_ij = tf.constant(np.zeros([self.n_inputs, 1152, 10, 1, 1], dtype=np.float32))

            # routing iterations
            for routing_iter in range(self.n_routing_iters):
                with tf.variable_scope('routing_iter' + str(routing_iter)):
                    # routing coefficients
                    c_ij = tf.nn.softmax(b_ij, dim=2)

                    # input to digit capsules
                    s_j = tf.multiply(c_ij, u_i)
                    s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)

                    # squash
                    s_j_norm = tf.norm(s_j, axis=3, keep_dims=True)
                    v_j = s_j_norm / (1 + s_j_norm) * s_j / tf.square(s_j_norm)

                    # correlate
                    correlations = tf.matmul(u_i, tf.tile(v_j, (1, 1152, 1, 1, 1)), transpose_a=True)
                    b_ij += correlations

        return v_j

    def cost(self):
        m_plus = tf.constant(0.90, dtype=tf.float32)
        m_minus = tf.constant(0.10, dtype=tf.float32)
        cost_lambda = tf.constant(0.50, dtype=tf.float32)

        d_caps_norm = tf.norm(self.d_capsules, axis=3)
        d_caps_norm = tf.squeeze(d_caps_norm, axis=3)

        total_cost = tf.reduce_sum(tf.multiply(self.Y, tf.square(tf.maximum(0.0, m_plus - d_caps_norm))), axis=2)
        assert total_cost.get_shape() == (self.n_inputs, 1)

        total_cost += cost_lambda * tf.reduce_sum(tf.multiply(1 - self.Y, tf.square(tf.maximum(0.0, d_caps_norm - m_minus))), axis=2)
        assert total_cost.get_shape() == (self.n_inputs, 1)

        total_cost = tf.reduce_sum(total_cost) / self.n_inputs
        return total_cost

    def fit(self, x, y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.X_in: x, self.Y_in: y})
        return cost








