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

        # TODO: check with
        # https://github.com/tensorflow/agents/blob/master/agents/ppo/utility.py#L122
        kl = tf.reduce_sum(tf.exp(self.old_var_ph - self.vars))
        kl += tf.reduce_sum(tf.square(self.means - self.old_mean_ph) /
                            tf.exp(self.vars), axis=1)
        kl += tf.reduce_sum(self.vars)
        kl -= tf.reduce_sum(self.old_var_ph)
        kl -= self.action_dim
        self.kl = .5 * tf.reduce_sum(kl)

        # TODO: check with
        # https://github.com/tensorflow/agents/blob/master/agents/ppo/utility.py#L139
        _ = self.action_dim * np.log(2 * np.pi * np.e)
        _ += tf.reduce_sum(self.vars)
        self.entropy = .5 * _

        _ = tf.exp(self.log_vars * .5)
        _ *= tf.random_normal(shape=(self.action_dim,))
        _ += self.means
        self.sampled = _

        loss_1 = -tf.reduce_mean(self.advantage_ph *
                                 tf.exp(self.log_prob - self.old_log_prob))
        loss_2 = tf.reduce_mean(self.beta_ph * self.kl)
        loss_3 = self.eta_ph * \
            tf.square(tf.maximum(0., self.kl - 2. * self.target_kl))
        self.loss = loss_1 + loss_2 + loss_3
        optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.train_op = optimizer.minimize(self.loss)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def policy(self, state):
        return self.session.run(self.sampled, {self.state_ph: state})

    def train(self, states, actions, advantages):
        feed_dict = {self.state_ph: states,
                     self.action_ph: actions,
                     self.advantage_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.learning_rate_ph: self.learning_rate * self.lr_coef}
        old_means, old_vars = self.session.run(
            [self.means, self.variances], feed_dict)

        feed_dict[self.old_mean_ph] = old_means
        feed_dict[self.old_vars] = old_vars

        for e in range(self.epochs):
            self.session.run(self.train_op, feed_dict)
            loss, kl, entropy = self.session.run(
                [self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.target_kl * 4:
                break
        if kl > self.target_kl * 2:
            self.beta = np.minimize(35, 1.5 * self.beta)
            if self.beta > 30 and self.lr_coef > .1:
                self.lr_coef /= 1.5
        elif kl < self.target_kl * .5:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)
            if self.beta < (1 / 30) and self.lr_coef < 10:
                self.lr_coef *= 1.5

    def close(self):
        self.session.close()
