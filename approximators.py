import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from numpy import ones
from tensorflow.contrib.framework import get_or_create_global_step
import numpy as np


def fullyConnected(name, input_layer, output_dims, activation=None, initializer=None, bias=True):

    if initializer is None or initializer == 'xavier':
        initializer = xavier_initializer()
    elif type(initializer) in [int, float]:
        initializer = tf.random_uniform_initializer(
            minval=-initializer, maxval=initializer)

    if type(input_layer) in [list, tuple]:
        print([[l.get_shape().as_list()[-1], output_dims]
               for i, l in enumerate(input_layer)])
        weights = [tf.get_variable("%s_w_%i" % (name, i), shape=[l.get_shape().as_list()[-1], output_dims],
                                   dtype=tf.float32, initializer=initializer) for i, l in enumerate(input_layer)]
        mults = [tf.matmul(l, w) for l, w in zip(input_layer, weights)]
        next_layer = sum(mults)

    else:
        input_dims = input_layer.get_shape().as_list()[1:]
        weight = tf.get_variable(name + "_w", shape=[*input_dims, output_dims],
                                 dtype=tf.float32, initializer=initializer)
        next_layer = tf.matmul(input_layer, weight)

    if bias:
        bias = tf.get_variable(
            name + "_b", shape=output_dims, dtype=tf.float32, initializer=initializer)
        next_layer = tf.add(next_layer, bias)

    if activation:
        next_layer = activation(next_layer, name=name + "_activated")

    return next_layer


def selectFromRows(tensor, indexes):
    shp = tf.shape(tensor)
    indexes_flat = tf.range(0, shp[0]) * shp[1] + indexes
    return tf.gather(tf.reshape(tensor, [-1]), indexes_flat)


def policyGradientLoss(action, policy_gradient):
    loss = tf.log(action) * policy_gradient
    loss = -tf.reduce_sum(loss)
    return loss


def entropyLoss(tensor):
    entropy = -tf.reduce_sum(tensor * tf.log(tensor), 1, name="entropy")
    return tf.reduce_mean(entropy, name="entropy_mean")


def getGradAndVars(optimizer, loss):
    grads_and_vars = optimizer.compute_gradients(loss)
    grads_and_vars = [[grad, var]
                      for grad, var in grads_and_vars if grad is not None]
    return grads_and_vars


def applyGradients(optimizer, from_grads, to_vars, clip_ratio=None):
    if clip_ratio:  # TODO: fix the error when True
        from_grads = tf.clip_by_global_norm(from_grads, clip_ratio)
    combined_grads_and_vars = zip(from_grads, to_vars)
    return optimizer.apply_gradients(combined_grads_and_vars)


def transferLearning(optimizer, from_loss, to_loss, clip_ratio=None):
    from_grads_and_vars = getGradAndVars(optimizer, from_loss)
    grads = [grad for grad, var in from_grads_and_vars]

    to_grads_and_vars = getGradAndVars(
        optimizer, to_loss)
    variables = [var for grad, var in to_grads_and_vars]

    return applyGradients(
        optimizer, grads, variables, clip_ratio)


def getScopeParameters(scope_name):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)


def copyScopeVars(from_scope, to_scope, tau=None):
    if tf.get_variable_scope().name:
        scope = tf.get_variable_scope().name + '/'
    else:
        scope = ''

    from_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + from_scope)
    target_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + to_scope)

    from_list = sorted(from_list, key=lambda v: v.name)
    target_list = sorted(target_list, key=lambda v: v.name)

    assert len(from_list) == len(target_list)
    assert len(target_list) > 0

    operations = []
    for i in range(len(from_list)):
        if tau is not None:
            new_value = tf.multiply(
                from_list[i], tau) + tf.multiply(target_list[i], (1 - tau))
        else:
            new_value = from_list[i]

        operations.append(target_list[i].assign(new_value))

    return operations


class Approximator(object):
    def __init__(self):
        self._loss = None
        pass

    def train(self, optimizer, other=None, clip_ratio=None):
        if other:
            assert type(other) == type(self), "Train source should be a %s object but it was %s object." % (
                self.__class__.__name__, other.__class__.__name__)
            assert other._loss is not None, "Loss of 'other' is not configured. Use '%s.loss()' method to configure it." % self.__class__.__name__
            assert self._loss is not None, "Loss of 'self' is not configured. Use '%s.loss()' method to configure it." % self.__class__.__name__
            return transferLearning(optimizer, other._loss, self._loss, clip_ratio)
        else:
            assert self._loss is not None, "Loss of 'self' is not configured. Use 'self.loss' method to configure it."
            if clip_ratio:
                return transferLearning(optimizer, self._loss, self._loss, clip_ratio)
            else:
                return optimizer.minimize(self._loss, global_step=get_or_create_global_step())


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


class Policy(Approximator):
    def __init__(self, input_layer, action_dims):
        super(Policy, self).__init__()
        with tf.variable_scope("policy_network"):
            self.action = fullyConnected(
                input_layer, action_dims, tf.nn.softmax)

    def loss(self, actions, policy_gradient):
        with tf.variable_scope("policy_network_loss"):
            actioned_probs = selectFromRows(self.action, actions)
            self._loss = policyGradientLoss(
                actioned_probs, policy_gradient)
        return self._loss


class Value(Approximator):
    def __init__(self, input_layer):
        super(Value, self).__init__()
        with tf.variable_scope("value_network"):
            self.value = fullyConnected("value", input_layer, 1)

    def loss(self, target):
        with tf.variable_scope("value_network_loss"):
            self._loss = tf.reduce_sum(tf.square(self.value - target))
        return self._loss


class ContinuousPolicy(Approximator):
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
