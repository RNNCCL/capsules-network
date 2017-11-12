import tensorflow as tf


class CapsuleNetwork(object):
    def __init__(self, n_inputs, routing_iters=3, optimizer=tf.train.AdamOptimizer(learning_rate=0.001)):
        # self.graph = tf.Graph()
        # with self.graph.as_default():

        self.n_inputs = n_inputs
        self.routing_iters = routing_iters

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
            print('conv dims: {}'.format(convs.get_shape()))

        with tf.variable_scope('PrimaryCapsules'):
            # 32 is number of capsule blocks
            p_capsules = tf.contrib.layers.conv2d(convs, num_outputs=8 * 32, kernel_size=9, stride=2,
                                                       padding='VALID', activation_fn=None)    # activation replaced by squashing function
            print('primary capsule layer dims: {}'.format(p_capsules.get_shape()))
            p_capsules = tf.reshape(p_capsules, (self.n_inputs, -1, 1, 8, 1))   # add dim for routing

        with tf.variable_scope('Routing'):
            w = tf.get_variable('weight_matrices', shape=(1, 1152, 10, 8, 16), dtype=tf.float32, initializer=tf.random_normal_initializer())

            u_i = tf.matmul(tf.tile(w, [self.n_inputs, 1, 1, 1, 1]), tf.tile(p_capsules, [1, 1, 10, 1, 1]), transpose_a=True)
            print('p_capsule output dims: {}'.format(u_i.get_shape()))

            # routing logits
            b_ij = tf.get_variable('routing_logits', shape=(1, 1152, 10, 1, 1), dtype=tf.float32,
                                   initializer=tf.zeros_initializer())

            # routing iterations
            for iter in range(self.routing_iters):
                with tf.variable_scope('routing_iter' + str(iter)):
                    # routing coefficients
                    c_ij = tf.nn.softmax(b_ij, dim=2)

                    # input to digit capsules
                    s_j = tf.multiply(tf.tile(c_ij, (self.n_inputs, 1, 1, 1, 1)), u_i)
                    s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)
                    # print('d_capsule input dims: {}'.format(s_j.get_shape()))

                    # digit capsule outputs
                    s_j_norm = tf.norm(s_j, axis=3, keep_dims=True)
                    v_j = s_j_norm / (1 + s_j_norm) * s_j / tf.square(s_j_norm)
                    # v_j = 1 - 1/s_j_norm * s_j / tf.square(s_j_norm)    # equivalent form
                    # print('d_capsule output dims: {}'.format(v_j.get_shape()))

                    # correlate p_capsule and d_capsule outputs
                    correlations = tf.matmul(u_i, tf.tile(v_j, (1, 1152, 1, 1, 1)), transpose_a=True)
                    b_ij += tf.reduce_sum(correlations, axis=0, keep_dims=True)

        return(v_j)

    def cost(self):
        m_plus = tf.constant(0.90, dtype=tf.float32)
        m_minus = tf.constant(0.10, dtype=tf.float32)
        cost_lambda = tf.constant(0.50, dtype=tf.float32)

        d_caps_norm = tf.norm(self.d_capsules, axis=3)
        d_caps_norm = tf.squeeze(d_caps_norm, axis=3)

        print('d_caps_norm dims {}, Y dims {}'.format(d_caps_norm.get_shape(), self.Y.get_shape()))

        total_cost = tf.reduce_sum(tf.multiply(self.Y, tf.square(tf.maximum(0.0, m_plus - d_caps_norm))), axis=2)
        assert total_cost.get_shape() == (self.n_inputs, 1)

        total_cost += cost_lambda * tf.reduce_sum(tf.multiply(1 - self.Y, tf.square(tf.maximum(0.0, d_caps_norm - m_minus))), axis=2)
        assert total_cost.get_shape() == (self.n_inputs, 1)

        total_cost = tf.reduce_sum(total_cost) / self.n_inputs
        print('total cost dim: {}'.format(total_cost.get_shape()))
        return total_cost

    def fit(self, X, Y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.X_in: X, self.Y_in: Y})
        return cost








