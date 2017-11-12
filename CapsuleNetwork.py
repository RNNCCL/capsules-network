import tensorflow as tf


class CapsuleNetwork(object):
    def __init__(self, n_inputs, optimizer=tf.train.AdamOptimizer()):
        # self.graph = tf.Graph()
        # with self.graph.as_default():

        self.n_inputs = n_inputs
        self.X = tf.placeholder(tf.float32, (self.n_inputs, 28, 28, 1))
        self.optimizer = optimizer

        # build network
        self.network()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def network(self):
        with tf.variable_scope('Conv'):
            conv_layer = tf.contrib.layers.conv2d(self.X, num_outputs=256, kernel_size=9, stride=1, padding='VALID')   # ReLU activation
            print('conv dims: {}'.format(conv_layer.get_shape()))

        with tf.variable_scope('PrimaryCapsule'):
            # 32 is number of capsule blocks
            primary_capsule_layer = tf.contrib.layers.conv2d(conv_layer, num_outputs=8 * 32, kernel_size=9, stride=2,
                                                       padding='VALID', activation_fn=None)    # activation replaced by squashing function
            print('primary capsule layer dims: {}'.format(primary_capsule_layer.get_shape()))
            primary_capsule_layer = tf.reshape(primary_capsule_layer, (self.n_inputs, 6, 6, 8, 32)) # last dimension are capsule blocks
            print("reshaped PC layer dims: {}".format(primary_capsule_layer.get_shape()))

        with tf.variable_scope('Routing'):
            W = tf.get_variable('weight_matrices', shape=(self.n_inputs, ))


