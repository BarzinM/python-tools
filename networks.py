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


def flatten(tensor):
    shape = tensor.get_shape().as_list()
    if len(shape) == 2:
        return tensor
    flat_size = np.prod(shape[1:], dtype=np.int32)
    print("Flatting tensor %s from %s to %s" %
          (tensor.name, str(shape), str([shape[0], flat_size])))
    return tf.reshape(tensor, [-1, flat_size])


def fanInStd(shape):
    if type(shape) == int:
        fan_in = shape
    elif type(shape) in [list, tuple]:
        fan_in = np.prod(shape[:-1])
    return 2. / np.sqrt(fan_in)


def fanIn(shape, mean=0.):
    std = fanInStd(shape)
    return tf.random_uniform_initializer(minval=-std + mean,
                                         maxval=std + mean)


def fullyConnected(name, flow, output_dims, activation, initializer=None, bias=True, return_param=False):
    assert type(name) == str
    if type(output_dims) not in [int, np.int16, np.int32, np.int64]:
        raise TypeError("The argument `output_dims` should be type `int`. "
                        "The provided value %s is type %s" % (str(output_dims), type(output_dims)))
    assert type(bias) == bool

    parameters = []

    if type(flow) in [list, tuple]:
        print("FULLY CONNECTING A LIST")
        mults = [fullyConnected(name='%s_%i' % (name, i),
                                flow=tensor,
                                output_dims=output_dims,
                                activation=None,
                                initializer=initializer,
                                bias=False) for i, tensor in enumerate(flow)]

        flow = tf.add_n(mults)

    else:
        flow = flatten(flow)
        dimension = flow.get_shape().as_list()[-1]

        if initializer is None or initializer == 'fanin':
            initializer = fanIn(dimension)
        elif initializer == 'xavier':
            initializer = xavier_initializer()
        elif type(initializer) in [int, float]:
            initializer = tf.random_uniform_initializer(minval=-initializer,
                                                        maxval=initializer)

        weight = tf.get_variable(name + "_weights",
                                 shape=[dimension, output_dims],
                                 dtype=tf.float32,
                                 initializer=initializer)
        parameters.append(weight)
        print("Fully connected weight:", name +
              "_w", weight.get_shape().as_list())
        flow = tf.matmul(flow, weight)

    if bias:
        bias = tf.get_variable(name + "_biases",
                               shape=[output_dims],
                               dtype=tf.float32,
                               initializer=initializer)
        parameters.append(bias)
        flow += bias

    if activation is not None:
        flow = activation(flow, name=name + "_activated")

    if return_param:
        parameters = [flow] + parameters
        return parameters
    else:
        return flow


class Convolutional(object):

    def __init__(self, max_pool=False):
        self.max_pool = max_pool
        self.padding = 'VALID'
        self.layers = []
        self.layer_dims = []
        self.generated = []
        self.vars = []
        self.deconv_vars = []

    def add(self, depth, patch_size, stride):
        if type(patch_size) == int:
            patch_size = [patch_size, patch_size]
        else:
            patch_size = list(patch_size)
        if type(stride) == int:
            stride = [stride, stride]
        else:
            stride = list(stride)
        self.layers.append((depth, patch_size, stride))

    def __call__(self, flow, dropout= None):
        self.batch_size = tf.shape(flow)[0]
        previous_depth = flow.get_shape().as_list()[-1]
        for i, (depth, patch_size, stride) in enumerate(self.layers):
            shape = patch_size + [previous_depth, depth]
            w = tf.get_variable("conv_w_%i" % i,
                                shape=shape,
                                initializer=fanIn(shape))
            b = tf.get_variable("conv_b_%i" % i,
                                shape=[depth],
                                initializer=fanIn(shape))
            self.vars.append((w, b, stride))
            self.generated.append([shape, stride])
            previous_depth = depth
        self.layers = []

        for w, b, stride in self.vars:
            self.layer_dims.append(flow.get_shape().as_list())

            stride = [1, *stride, 1]

            if self.max_pool:
                flow = tf.nn.conv2d(
                    flow, w, strides=[1, 1, 1, 1], padding=self.padding)
                flow = tf.nn.max_pool(
                    flow, stride, stride, padding=self.padding)
            else:
                flow = tf.nn.conv2d(flow, w, strides=stride,
                                    padding=self.padding)

            flow = tf.nn.relu(flow + b)

            if dropout is not None:
                flow = tf.nn.dropout(flow, keep_prob=dropout)

        return flow

    def transpose(self, flow, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # for i in range(len(self.depth)):
        #     next_shape = [-(-self._shapes[i][0] // self.stride[0]), -
        #                   (-self._shapes[i][1] // self.stride[1])]
        #     self._shapes.append(next_shape)

        flow_shape = flow.get_shape().as_list()
        # assert self.layer_dims[-1] == flow_shape, "%s,%s do not match" % (
        #     self.layer_dims[-1], flow_shape)

        print(self.generated)
        print(list(zip(self.generated, self.layer_dims)))
        layer_info = reversed(list(zip(self.generated, self.layer_dims)))
        for i, ((shape, stride), dims) in enumerate(layer_info):
            # if i == len(self.generated)-1:
            #     shape[3] = flow_shape[3]

            w = tf.get_variable("deconv_w_%i" % i, shape=shape)
            b = tf.get_variable("deconv_b_%i" % i, shape=shape[2])
            self.deconv_vars.append((w, b, stride, dims))

        for i, (w, b, stride, dims) in enumerate(self.deconv_vars):
            print(flow)
            print(w)
            print(dims)
            dims[0] = batch_size
            flow = tf.nn.conv2d_transpose(flow, w, output_shape=dims, strides=[1, *stride, 1], padding="VALID")
            if i < len(self.deconv_vars) - 1:
                flow = tf.nn.relu(flow + b)
            print('---------')

        return flow

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
                flow = tf.nn.bias_add(flow, b, 'NHWC')
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


class BaseNetwork(object):

    def __init__(self):
        # self.drop = tf.placeholder(tf.float32, name="dropout")
        # self.normalize = tf.placeholder(tf.bool, name="normalize")
        self.global_step = tf.train.get_or_create_global_step()
        # learning_rate = LEARNING_RATE
        # self.learning_rate = tf.train.exponential_decay(
        #     LEARNING_RATE, self.global_step, DECAY_STEP, LEARNING_DECAY)

    def initSaver(self):
        # self.save_ready = True
        self.saver = tf.train.Saver(max_to_keep=40)

    def initHistogram(self, size, numbers):
        self.counter = 0
        self.histo = np.empty((size, numbers))

    def store(self, *args):
        self.histo[self.counter] = args
        self.counter += 1

    def saveHistogram(self, path):
        np.save(path, self.histo)

    def saveModel(self, session, path, step=0):
        path = self.saver.save(
            session, path, global_step=step or self.global_step)
        print("Saved model to", path)

    def restoreModel(self, session, path):
        self.saver.restore(session, path)
