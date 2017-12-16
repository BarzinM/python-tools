import tensorflow as tf


class BaseAgent(object):

    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build([100], [100])
        self.session = tf.Session(graph=self.graph)
        self.session.run(tf.global_variables_initializer())

    def _build(self):
        self.reward_ph = tf.placeholder(tf.float32, (None,), 'rewards')
        self.terminal_ph = tf.placeholder(tf.bool, (None,), 'terminals')
        self.global_step = tf.train.get_or_create_global_step()
