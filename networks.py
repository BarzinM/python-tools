import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers.python.layers import batch_norm

import numpy as np


def normalizeBatch(x, is_training, name):
    return batch_norm(x, decay=0.9, center=True, scale=True,
                      updates_collections=None,
                      is_training=is_training,
                      reuse=None,
                      trainable=True,
                      scope=name)


def fullyConnected(name, input_layer, output_dims, activation=None, initializer=None, bias=True):

    if initializer is None or initializer == 'xavier':
        initializer = xavier_initializer()
    elif type(initializer) in [int, float]:
        initializer = tf.random_uniform_initializer(
            minval=-initializer, maxval=initializer)

    if type(input_layer) in [list, tuple]:
        print("fully connected shapes", [[l.get_shape().as_list()[-1], output_dims]
                                         for i, l in enumerate(input_layer)])
        weights = [tf.get_variable("%s_w_%i" % (name, i), shape=[l.get_shape().as_list()[-1], output_dims],
                                   dtype=tf.float32, initializer=initializer) for i, l in enumerate(input_layer)]
        mults = [tf.matmul(l, w) for l, w in zip(input_layer, weights)]
        next_layer = tf.add_n(mults)

    else:
        input_dims = input_layer.get_shape().as_list()[1:]
        weight = tf.get_variable(name + "_w", shape=input_dims + [output_dims],
                                 dtype=tf.float32, initializer=initializer)
        next_layer = tf.matmul(input_layer, weight)

    if bias:
        bias = tf.get_variable(
            name + "_b", shape=output_dims, dtype=tf.float32, initializer=initializer)
        next_layer = tf.add(next_layer, bias)

    if activation:
        next_layer = activation(next_layer, name=name + "_activated")

    return next_layer


class Conv(object):

    def __init__(self, depth, stride, patch, padding):
        self.padding = padding
        self.depth = list(depth)
        self.stride = list(stride)
        self.patch = list(patch)
        self._shapes = []
        self.conv_params = []

    def encode(self, flow, dropout=None, normalize=None, initializer=None, activation=tf.nn.relu, bias=True):
        # self._shapes.append(flow.get_shape().as_list()[1:-1])
        if initializer is not None:
            initializer = tf.random_uniform_initializer(
                minval=-initializer, maxval=initializer)
        else:
            initializer = None
        channels = flow.get_shape()[3]
        self.depth = [channels] + self.depth

        for i in range(len(self.depth[:-1])):
            w = tf.get_variable("conv_weight_%i" % i, shape=list(self.patch) +
                                [self.depth[i], self.depth[i + 1]], initializer=initializer, dtype=tf.float32)
            if bias:
                b = tf.Variable(tf.constant(.01, shape=[self.depth[i + 1]]))
            else:
                b = None
            self.conv_params.append((w, b))

        if normalize is not None:
            flow = normalizeBatch(flow, normalize, 'normalize_input')
        if dropout is not None:
            flow = tf.nn.dropout(flow, keep_prob=dropout)

        i = 0
        self.layers = [flow]
        for w, b in self.conv_params:
            flow = tf.nn.conv2d(
                flow, w, [1, 1, 1, 1], padding=self.padding)
            if bias:
                flow += b
            self._shapes.append(flow.get_shape().as_list()[1:-1])
            flow = tf.nn.max_pool(
                flow, [1] + self.stride + [1], [1] + self.stride + [1], padding=self.padding)
            flow = activation(flow)

            if normalize is not None:
                flow = normalizeBatch(
                    flow, normalize, 'normalize_layer_%i' % i)
            if dropout is not None:
                flow = tf.nn.dropout(flow, keep_prob=dropout)
            self.layers.append(flow)
            i += 1

        print("Conv shapes", self._shapes)
        self.shape = flow.get_shape().as_list()
        self.encoded = flow

        return flow

    def l2(self):
        return tf.add_n([tf.nn.l2_loss(w) for w, b in self.conv_params])

    def l1(self):
        return tf.add_n([tf.reduce_sum(tf.abs(w)) for w, b in self.conv_params])

    def info(self):
        print("Conv layers dimensions:", self._shapes)

    def flat(self):
        size = self.shape[1] * self.shape[2] * self.shape[3]
        return tf.reshape(self.encoded, (-1, size))

    def decode(self, flow):
        # for i in range(len(self.depth)):
        #     next_shape = [-(-self._shapes[i][0] // self.stride[0]), -
        #                   (-self._shapes[i][1] // self.stride[1])]
        #     self._shapes.append(next_shape)

        self._shapes = list(reversed(self._shapes))
        deconv_params = []
        for i in range(len(self.depth[:-1])):
            w = tf.get_variable("deconv_weight_%i" % i, shape=self.patch + [
                self.depth[i], self.depth[i + 1]], initializer=self.initializer, dtype=tf.float32)
            b = tf.Variable(tf.constant(.01, shape=[self.depth[i]]))
            deconv_params.append((w, b))
        deconv_params = list(reversed(deconv_params))
        shape = flow.get_shape().as_list()
        batch_size, height, width, self.depth = shape
        batch_size = tf.shape(flow)[0]
        for i, p in enumerate(deconv_params):
            w, b = p
            next_depth, depth = w.get_shape().as_list()[-2:]
            height, width = self._shapes[i + 2]

            flow = tf.nn.conv2d_transpose(flow, w, strides=[1] + self.stride + [1], output_shape=[
                                          batch_size, height, width, next_depth], padding="SAME")
            flow = self.activation(flow)

        return flow


class RandomNetwork(object):

    def __init__(self, dimensions):
        self.noise = tf.placeholder(tf.float32, (None, 1))
        self.output = tf.placeholder(tf.float32, (None, dimensions))
        self.predict = fullyConnected("nonlinearities", self.noise, dimensions)
        self.loss = tf.reduce_mean(tf.square(self.output - self.predict))
        optimizer = tf.train.AdamOptimizer(.001)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, session, data):
        noise = np.random.rand(data.shape[0], 1)
        return session.run([self.loss, self.train_op], {
            self.output: data,
            self.noise: noise
        })[0]


class ContinuousPolicy(object):

    def __init__(self, input_layer, action_dims, bounds):
        super(ContinuousPolicy, self).__init__()
        assert len(bounds) == 2, "Bound should be list/tuple with two elements."
        assert bounds[0] < bounds[
            1], "The first bound value should be smaller than the second one."

        with tf.variable_scope("policy_network"):
            # with tf.variable_scope("mu"):
            mu = fullyConnected("mu", input_layer, action_dims)
            # with tf.variable_scope("sigma"):
            sigma = fullyConnected("sigma", input_layer, action_dims)
            sigma = tf.nn.softplus(sigma)
            self.distribution = tf.contrib.distributions.Normal(mu, sigma)
            action = self.distribution.sample_n(1)
            self.action = tf.clip_by_value(action, bounds[0], bounds[1])

    def loss(self, action, policy_gradient):
        with tf.variable_scope("policy_network_loss"):
            self._loss = -self.distribution.log_prob(action) * policy_gradient
        return self._loss

    def entropy(self):
        return self.distribution.entropy()
