import tensorflow as tf
from base_agent import BaseAgent
from networks import fullyConnected
import numpy as np


class PPO(BaseAgent):
    def __init__(self, state_dim, action_dim, target_kl):
        self.beta = 1.
        self.eta = 50
        self.target_kl = target_kl
        self.epochs = 20
        self.lr_coef = 1.
        self.learning_rate = .0001
        self.initial_var = 1.

        super().__init__(self, state_dim, action_dim)
        self.action_ph = tf.placeholder(
            tf.float32, (None, action_dim), 'actions')
        self.advantage_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        self.beta_ph = tf.placeholder(tf.float32, name='beta')
        self.eta_ph = tf.placeholder(tf.float32, name='eta')

        self.learning_rate_ph = tf.placeholder(
            tf.float32, name='learning_rate')

        self.old_var_ph = tf.placeholder(
            tf.flaot32, (action_dim,), 'old_vars')  # TODO: is dimension correct?
        self.old_mean_ph = tf.placeholder(
            tf.float32, (None, self.action_dim), 'old_means')

    def build(self, dimensions):
        assert type(dimensions) in [list, tuple]
        flow = self.state_ph
        for i, dim in enumerate(dimensions):
            # TODO: PPO with tanh or relu
            flow = fullyConnected("layer_%i" % i, flow, dim, tf.nn.relu)
        self.means = fullyConnected("means", flow, self.action_dim, None)
        self.vars = tf.get_variable('vars',
                                    (self.action_dim,),
                                    tf.float32,
                                    tf.constant_initializer(self.initial_var))

        def _temp(means, variances):
            log_prob = tf.reduce_sum(tf.square(self.action_ph - means))
            log_prob /= tf.exp(variances, axis=1)  # TODO: need axis?
            log_prob += tf.reduce_sum(variances)
            log_prob *= -.5
            return log_prob

        self.log_prob = _temp(self.means, self.vars)
        self.old_log_prob = _temp(self.old_mean_ph, self.old_var_ph)

        # TODO: check with https://github.com/tensorflow/agents/blob/master/agents/ppo/utility.py#L122
        kl = tf.reduce_sum(tf.exp(self.old_var_ph - self.vars))
        kl += tf.reduce_sum(tf.square(self.means - self.old_mean_ph) /
                            tf.exp(self.vars), axis=1)
        kl += tf.reduce_sum(self.vars)
        kl -= tf.reduce_sum(self.old_var_ph)
        kl -= self.action_dim
        self.kl = .5 * tf.reduce_sum(kl)

        # TODO: check with https://github.com/tensorflow/agents/blob/master/agents/ppo/utility.py#L139
        entropy = self.action_dim * np.log(2 * np.pi * np.e)
        entropy += tf.reduce_sum(self.vars)
        self.entropy = entropy * .5

        
