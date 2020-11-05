import tensorflow as tf
#from tensorflow.python import control_flow_ops
from tensorflow.python.ops import control_flow_ops

class BatchNorm():
    def __init__(self,
                 input,
                 training,
                 decay=0.95,
                 epsilon=1e-4,
                 name='bn',
                 reuse_vars=False):
        """
        Initialize the model.

        Args:
            self: (todo): write your description
            input: (str): write your description
            training: (todo): write your description
            decay: (float): write your description
            epsilon: (float): write your description
            name: (str): write your description
            reuse_vars: (list): write your description
        """

        self.decay = decay
        self.epsilon = epsilon
        self.batchnorm(input, training, name, reuse_vars)

    def batchnorm(self, input, training, name, reuse_vars):
        """
        Batch normalization.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            training: (todo): write your description
            name: (str): write your description
            reuse_vars: (bool): write your description
        """
        with tf.variable_scope(name, reuse=reuse_vars) as bn:
            rank = len(input.get_shape().as_list())
            in_dim = input.get_shape().as_list()[-1]

            if rank == 2:
                self.axes = [0]
            elif rank == 4:
                self.axes = [0, 1, 2]
            else:
                raise ValueError('Input tensor must have rank 2 or 4.')

            self.offset = tf.get_variable(
                'offset',
                shape=[in_dim],
                initializer=tf.constant_initializer(0.0))

            self.scale = tf.get_variable(
                'scale',
                shape=[in_dim],
                initializer=tf.constant_initializer(1.0))

            self.ema = tf.train.ExponentialMovingAverage(decay=self.decay)

            self.output = tf.cond(training,
                                  lambda: self.get_normalizer(input, True),
                                  lambda: self.get_normalizer(input, False))

    def get_normalizer(self, input, train_flag):
        """
        Get normalizer.

        Args:
            self: (todo): write your description
            input: (str): write your description
            train_flag: (str): write your description
        """
        if train_flag:
            self.mean, self.variance = tf.nn.moments(input, self.axes)
            # Fixes numerical instability if variance ~= 0, and it goes negative
            v = tf.nn.relu(self.variance)
            ema_apply_op = self.ema.apply([self.mean, self.variance])
            with tf.control_dependencies([ema_apply_op]):
                self.output_training = tf.nn.batch_normalization(
                    input, self.mean, v, self.offset, self.scale,
                    self.epsilon, 'normalizer_train'),
            return self.output_training
        else:
            self.output_test = tf.nn.batch_normalization(
                input, self.ema.average(self.mean),
                self.ema.average(self.variance), self.offset, self.scale,
                self.epsilon, 'normalizer_test')
            return self.output_test

    def get_batch_moments(self):
        """
        Gets the moments.

        Args:
            self: (todo): write your description
        """
        return self.mean, self.variance

    def get_ema_moments(self):
        """
        Compute the moments.

        Args:
            self: (todo): write your description
        """
        return self.ema.average(self.mean), self.ema.average(self.variance)

    def get_offset_scale(self):
        """
        Get the offset of the scale

        Args:
            self: (todo): write your description
        """
        return self.offset, self.scale
