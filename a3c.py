import numpy as np
import tensorflow as tf
from approximators import fullyConnected, copyScopeVars, getScopeParameters, entropyLoss
from memory import Memory, GeneralMemory
from stats import RunningAverage
from collections import deque
from time import time



class A3C(object):
    def __init__(self, state_dim, action_dim, memory_size, shared_net=[]):
        self.action_dim = action_dim
        self.state = tf.placeholder(tf.float32, (None, state_dim))
        self.shared = self.state
        with tf.variable_scope("shared"):
            for i, layer in enumerate(shared_net):
                self.shared = fullyConnected(
                    "layer_%i" % i, self.shared, layer, tf.nn.relu)
        self.action_ph = tf.placeholder(tf.int32, (None))
        self.grad_ph = tf.placeholder(tf.float32, (None, 1))
        self.value_ph = tf.placeholder(tf.float32, (None, 1))
        self.memory = Memory(memory_size, state_dim, 0)

    def initializeCritic(self, layers=[400, 300], optimizer=None):
        optimizer = optimizer or tf.train.AdamOptimizer(.01)

        def _make():
            net = self.shared
            for i, layer in enumerate(layers):
                net = fullyConnected("layer_%i" %
                                     i, net, layer, tf.nn.relu, .1)
            return fullyConnected("critic_output", net, 1, initializer=.001)

        with tf.variable_scope("critic"):
            self.critic = _make()

        self.critic_loss = tf.reduce_mean(
            tf.square(self.value_ph - self.critic))
        self.train_critic = optimizer.minimize(self.critic_loss)

    def initializeActor(self, layers=[400, 300], optimizer=None, entropy=0.0):
        optimizer = optimizer or tf.train.AdamOptimizer(.0001)

        def _make():
            net = self.shared
            for i, layer in enumerate(layers):
                net = fullyConnected("layer_%i" %
                                     i, net, layer, tf.nn.relu, .01)
            actions = fullyConnected(
                "actor_output", net, self.action_dim, tf.nn.softmax, .01)
            return actions

        with tf.variable_scope("actor"):
            self.actor = _make()

        row = tf.expand_dims(tf.range(0, tf.shape(self.actor)[0]), axis=1)
        col = tf.expand_dims(self.action_ph, axis=1)
        indexes = tf.concat([row, col], axis=1)
        action_probibility = tf.gather_nd(self.actor, indexes)

        self.single_loss = tf.log(action_probibility + 1e-8) * self.grad_ph
        self.actor_loss = -tf.reduce_sum(self.single_loss)
        if entropy != 0.0:
            self.actor_loss -= entropyLoss(self.actor) * entropy

        self.train_actor = optimizer.minimize(self.actor_loss)

    def train(self, session, batch_size, discount=.97):
        states, actions, rewards, next_states, terminals = self.memory.sample(
            batch_size)
        next_state_value = session.run(
            self.critic, {self.state: next_states})
        state_value = session.run(
            self.critic, {self.state: states})
        observed_value = rewards + discount * next_state_value
        observed_value[terminals] = rewards[terminals] / (1 - discount)

        l, _ = session.run([self.critic_loss, self.train_critic], {
                           self.value_ph: observed_value, self.state: states})
        _, actor_l = session.run([self.train_actor, self.actor_loss], {
            self.state: states, self.action_ph: actions, self.grad_ph: observed_value - state_value})

        return l

    def policy(self, session, state):
        a = session.run(self.actor, {self.state: [state]})[0]
        return a

    def memorize(self, state, action, reward, next_state, terminal):
        self.memory.add(state, action, reward, next_state, terminal)
