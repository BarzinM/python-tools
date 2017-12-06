import tensorflow as tf


class BaseAgent(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_ph = tf.placeholder(tf.float32, (None, state_dim), 'states')
