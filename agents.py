import numpy as np
import tensorflow as tf
from approximators import fullyConnected
from memory import Memory


def choice(probabilities):
    return np.random.choice(len(probabilities), p=probabilities)


def epsGreedy(probabilities, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(probabilities))
    else:
        return np.argmax(probabilities)


def greedy(probabilities):
    return np.argmax(probabilities)


class DQN(object):
    def __init__(self, state_dim, action_dim, memory_size):
        self.action_dim = action_dim
        self.state_ph = tf.placeholder(tf.float32, [None, state_dim])
        self.action_ph = tf.placeholder(tf.float32, [None, action_dim])
        self.memory = Memory(memory_size, state_dim, 0)

    def initialize(self, layer_dims=[100], optimizer=None):
        flow = self.state_ph
        for i, size in enumerate(layer_dims):
            flow = fullyConnected("layer%i" % i, flow, size, tf.nn.relu)

        self.value = fullyConnected("output_layer", flow, self.action_dim)
        self._loss = tf.reduce_mean(tf.square(self.value - self.action_ph))

        self.optimizer = optimizer or tf.train.AdamOptimizer(.01)
        self.train_op = self.optimizer.minimize(self._loss)

    def train(self, session, batch=None, discount=.97):
        if type(batch) == int:
            states, actions, rewards, next_states, terminals = self.memory.sample(
                batch)
            action_value = session.run(
                self.value, {self.state_ph: states})
            next_state_value = session.run(
                self.value, {self.state_ph: next_states})
            observed_value = rewards + discount * \
                np.max(next_state_value, 1, keepdims=True)
            observed_value[terminals] = rewards[terminals] / (1 - discount)
            action_value[np.arange(batch), actions] = observed_value[:, 0]

        elif type(batch) in [list, tuple]:
            states, action_value = batch

        _, l = session.run([self.train_op, self._loss], {
            self.state_ph: states, self.action_ph: action_value})
        return l

    def policy(self, session, state):
        return session.run(self.value, {self.state_ph: [state]})[0]

    def memorize(self, state, action, reward, next_state, terminal):
        self.memory.add(state, action, reward, next_state, terminal)
