import tensorflow as tf
from base_agent import BaseAgent
from networks import fullyConnected
import numpy as np


class ValueFunction(object):

    def __init__(self, state_dim):
        self.state_dim = state_dim

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build()
        self.session = tf.Session(graph=self.graph)
        self.session.run(tf.global_variables_initializer())

    def _build(self):
        self.state_ph = tf.placeholder(
            tf.float32, (None, state_dim), 'states')
        self.target_values = tf.placeholder(tf.float32(None,), 'values')
        flow = self.state_ph
        for i, dim in enumerate(dimensions):
            flow = fullyConnected('layer_%i' % i, flow, dim, tf.nn.relu)
        self.value = fullyConnected('output', flow, 1, None)
        self.loss = tf.reduce_mean(tf.square(self.value - self.target_values))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def fit(self, x, y):
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat) / np.var(y) # what is this used for?
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run(
                    [self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        # explained variance after update
        loss = np.mean(np.square(y_hat - y))
        # diagnose over-fitting of val func
        exp_var = 1 - np.var(y - y_hat) / np.var(y)

    def predict(self, x):
        prediction = self.session.run(self.value, {self.state_ph: x})
        return np.squeeze(prediction)

    def close(self):
        self.session.close()
