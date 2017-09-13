import numpy as np
import tensorflow as tf
from tfmisc import copyScopeVars, getScopeParameters, entropyLoss
from networks import fullyConnected
from memory import Memory
from stats import RunningAverage
from collections import deque


class DQN(object):

    def __init__(self, state_dim, action_dim, memory_size):
        self.action_dim = action_dim
        if type(state_dim) == int:
            self.state = tf.placeholder(
                tf.float32, [None, state_dim], "states")
        if type(state_dim) in [list, tuple]:
            self.state = tf.placeholder(tf.float32, [None, *state_dim], "states")

        self.action_ph = tf.placeholder(tf.int32, [None], "actions")
        self.action_value_ph = tf.placeholder(
            tf.float32, [None], "action_values")
        self.memory = Memory(
            memory_size, state_dim, 0, 1, state_dim, -1)

    def initialize(self, layer_dims, optimizer, learner_target_inputs=None):
        def _make(flow):
            for i, size in enumerate(layer_dims):
                flow = fullyConnected(
                    "layer%i" % i, flow, size, tf.nn.relu, initializer=.003)

            return fullyConnected(
                "output_layer", flow, self.action_dim, initializer=.003)

        learner_target_inputs = learner_target_inputs or [
            self.state, self.state]
        with tf.variable_scope('learner'):
            self.action_value = _make(learner_target_inputs[0])
        with tf.variable_scope('target'):
            self.target_action_value = _make(learner_target_inputs[1])

        self.update_op = copyScopeVars('learner', 'target')

        row = tf.range(0, tf.shape(self.action_value)[0])
        indexes = tf.stack([row, self.action_ph], axis=1)
        action_value = tf.gather_nd(self.action_value, indexes)

        self.single_loss = tf.square(action_value - self.action_value_ph)
        self._loss = tf.reduce_mean(self.single_loss)

        self.train_op = optimizer.minimize(self._loss)

    def train(self, session, batch=None, discount=.97):
        states, actions, rewards, next_states, terminals = self.memory.sample(
            batch)
        next_state_value = session.run(
            self.target_action_value, {self.state: next_states})
        observed_value = rewards + discount * \
            np.max(next_state_value, 1, keepdims=True)
        observed_value[terminals] = rewards[terminals]

        _, l = session.run([self.train_op, self._loss], {
            self.state: states, self.action_ph: actions, self.action_value_ph: observed_value[:, 0]})
        return l

    def policy(self, session, state):
        return session.run(self.action_value, {self.state: [state]})[0]

    def memorize(self, state, action, reward, next_state, terminal):
        self.memory.add(state, action, reward, next_state, terminal)

    def update(self, session):
        session.run(self.update_op)


class DuelingDQN(DQN):

    def initialize(self, layer_dims=[100], optimizer=None):
        def _make():
            flow = self.state
            for i, size in enumerate(layer_dims):
                flow = fullyConnected("layer%i" % i, flow, size, tf.nn.relu)

            value = fullyConnected(
                "output_layer", flow, 1)
            advantage = fullyConnected("advantage", flow, self.action_dim)
            return value + advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)

        with tf.variable_scope('learner'):
            self.action_value = _make()

        with tf.variable_scope('target'):
            self.target_action_value = _make()

        self.update_op = copyScopeVars('learner', 'target')

        row = tf.range(0, tf.shape(self.action_value)[0])
        indexes = tf.stack([row, self.action_ph], axis=1)
        action_value = tf.gather_nd(self.action_value, indexes)

        self._loss = tf.reduce_mean(
            tf.square(action_value - self.action_value_ph))

        self.train_op = optimizer.minimize(self._loss)


class DoubleDQN(DQN):

    def train(self, session, batch=None, discount=.97):
        states, actions, rewards, next_states, terminals = self.memory.sample(
            batch)
        target_av, action_value = session.run(
            [self.target_action_value, self.action_value], {self.state: next_states})
        next_action = np.argmax(action_value, axis=1)
        observed_value = rewards[:, 0] + discount * \
            target_av[np.arange(batch), next_action]
        observed_value[terminals] = rewards[terminals, 0]

        _, l = session.run([self.train_op, self._loss], {
            self.state: states, self.action_value_ph: observed_value})
        return l


class DoubleDuelingDQN(DuelingDQN, DoubleDQN):
    pass


class ExpDQN(object):

    def __init__(self, state_dim, action_dim, memory_size):
        self.action_dim = action_dim
        self.state = tf.placeholder(tf.float32, [None, state_dim])
        self.action_ph = tf.placeholder(tf.int32, [None])
        self.action_value_ph = tf.placeholder(tf.float32, [None])

        self.memory = Memory(
            memory_size, state_dim, 0, 1, state_dim, -1)
        self.com_memory = Memory(
            memory_size, state_dim, 0, 1, state_dim, -1)

        self.com_average = RunningAverage(.001)
        self.mem_average = RunningAverage(.001)

        self.q = deque(maxlen=1)

    def initialize(self, layer_dims=[100], optimizer=None):
        def _make():
            flow = self.state
            for i, size in enumerate(layer_dims):
                flow = fullyConnected("layer%i" % i, flow, size, tf.nn.relu)

            return fullyConnected(
                "output_layer", flow, self.action_dim)

        with tf.variable_scope('learner'):
            self.action_value = _make()
        with tf.variable_scope('target'):
            self.target_action_value = _make()

        self.update_op = copyScopeVars('learner', 'target')

        row = tf.expand_dims(
            tf.range(0, tf.shape(self.action_value)[0]), axis=1)
        col = tf.expand_dims(self.action_ph, axis=1)
        indexes = tf.concat([row, col], axis=1)
        action_value = tf.gather_nd(self.action_value, indexes)

        self.single_loss = tf.square(action_value - self.action_value_ph)
        self._loss = tf.reduce_mean(self.single_loss)

        self.train_op = optimizer.minimize(self._loss)

    def _train(self, session, memory, batch, discount):
        states, actions, rewards, next_states, terminals = memory.sample(
            batch)
        next_state_value = session.run(
            self.target_action_value, {self.state: next_states})
        observed_value = rewards + discount * \
            np.max(next_state_value, 1, keepdims=True)
        observed_value[terminals] = rewards[terminals]
        return states, actions, observed_value[:, 0]

    def train(self, session, batch=None, discount=.97, p=False):
        s, a, av = self._train(session, self.memory, batch, discount)

        try:
            cs, ca, cav = self._train(session, self.com_memory, 2, discount)
        except ValueError:
            self.com_memory.add(*self.memory[0])
            cs, ca, cav = self._train(session, self.com_memory, 2, discount)

        s = np.concatenate([s, cs])
        a = np.concatenate([a, ca])
        av = np.concatenate([av, cav])

        _, loss = session.run([self.train_op, self.single_loss], {
            self.state: s, self.action_ph: a, self.action_value_ph: av})
        index = loss[:batch] > self.com_average.avg

        if any(index):
            memory_index = self.memory.getBatchIndex()[index]
            self.q.append(memory_index[0])

        self.com_average.update(np.mean(loss[batch:]))
        self.mem_average.update(np.mean(loss[:batch]))
        return np.mean(loss)

    def policy(self, session, state):
        return session.run(self.action_value, {self.state: [state]})[0]

    def memorize(self, session, state, action, reward, next_state, terminal):
        next_state_value = session.run(
            self.target_action_value, {self.state: [next_state]})[0]
        if terminal:
            observed_value = reward / .03
        else:
            observed_value = reward + .97 * np.max(next_state_value)
        loss = observed_value - \
            session.run(self.target_action_value, {
                        self.state: [state]})[0][action]
        loss = np.square(loss)
        if loss < self.com_average.avg or self.memory.count() == 0:
            if self.q:
                i = self.q.pop()
                self.com_memory.add(*self.memory[i])
                self.memory[i] = state, action, reward, next_state, terminal
            else:
                self.memory.add(state, action, reward, next_state, terminal)
            return 0, loss
        else:
            self.com_memory.add(state, action, reward, next_state, terminal)
            return 1, loss

    def update(self, session):
        session.run(self.update_op)


class ExpDQN2(object):

    def __init__(self, state_dim, action_dim, memory_size):
        self.action_dim = action_dim
        self.state = tf.placeholder(tf.float32, [None, state_dim], "states")
        self.action_ph = tf.placeholder(tf.int32, [None], "actions")
        self.action_value_ph = tf.placeholder(
            tf.float32, [None], "action_values")
        self.memory = Memory(
            memory_size, state_dim, 0, 1, state_dim, -1)
        self.com_memory = Memory(
            memory_size, state_dim, 0, 1, state_dim, -1)

        self.com_avg = RunningAverage(.001, 1.)
        self.mem_avg = RunningAverage(.001, 0.)

    def initialize(self, layer_dims=[100], optimizer=None):
        def _make():
            flow = self.state
            for i, size in enumerate(layer_dims):
                flow = fullyConnected(
                    "layer%i" % i, flow, size, tf.nn.relu, initializer=.003)

            return fullyConnected(
                "output_layer", flow, self.action_dim, initializer=.003)

        with tf.variable_scope('learner'):
            self.action_value = _make()
        with tf.variable_scope('target'):
            self.target_action_value = _make()

        self.update_op = copyScopeVars('learner', 'target')

        row = tf.range(0, tf.shape(self.action_value)[0])
        indexes = tf.stack([row, self.action_ph], axis=1)
        action_value = tf.gather_nd(self.action_value, indexes)

        self.single_loss = tf.square(action_value - self.action_value_ph)
        self.get_grad = tf.gradients(self.single_loss, self.action_value)[0]
        up, down = tf.split(self.get_grad, 2, 0)
        min_grad = tf.reduce_mean(tf.abs(up))
        down = tf.clip_by_value(down, -min_grad, min_grad)
        grads = tf.concat([up, down], 0)
        parameters = getScopeParameters("learner")
        grads = tf.gradients(self.action_value, parameters, grads)
        # self._loss = tf.reduce_mean(self.single_loss)
        clipped = zip(grads, parameters)
        self.train_op = optimizer.apply_gradients(
            clipped)  # self.optimizer.minimize(self._loss)

    def train(self, session, batch=None, discount=.97):
        states, actions, rewards, next_states, terminals = self.memory.sample(
            batch)
        next_state_value = session.run(
            self.target_action_value, {self.state: next_states})
        observed_value = rewards + discount * \
            np.max(next_state_value, 1, keepdims=True)
        observed_value[terminals] = rewards[terminals]

        try:
            states2, actions2, rewards2, next_states2, terminals2 = self.com_memory.sample(
                batch)
            next_state_value2 = session.run(
                self.target_action_value, {self.state: next_states2})
            observed_value2 = rewards2 + discount * \
                np.max(next_state_value2, 1, keepdims=True)
            observed_value2[terminals2] = rewards2[terminals2]

            states = np.concatenate((states, states2), 0)
            actions = np.concatenate((actions, actions2), 0)
            observed_value = np.concatenate(
                (observed_value, observed_value2), 0)
        except ValueError:
            pass

        _, sl = session.run([self.train_op, self.single_loss], {
            self.state: states, self.action_ph: actions, self.action_value_ph: observed_value[:, 0]})

        if len(sl) > batch:
            self.com_avg.update(np.max(sl[batch:]))
        # self.com_avg.update(np.mean(sl))

        return np.mean(sl[:batch])

    def policy(self, session, state):
        return session.run(self.action_value, {self.state: [state]})[0]

    def memorize(self, session, state, action, reward, next_state, terminal):
        # if not self.memory.isFull():
        #     self.memory.add(state, action, reward, next_state, terminal)
        #     return
        next_state_value = session.run(
            self.target_action_value, {self.state: [next_state]})
        observed_value = reward + .97 * \
            np.max(next_state_value, 1, keepdims=True)
        if terminal:
            observed_value[0, 0] = reward / .03
        l = session.run(self.single_loss, {
            self.state: state[None, :], self.action_ph: [action], self.action_value_ph: observed_value[:, 0]})[0]

        if l > self.com_avg.avg:
            self.com_memory.add(state, action, reward, next_state, terminal)
        else:
            self.memory.add(state, action, reward, next_state, terminal)
            # print("skipping")

    def update(self, session):
        session.run(self.update_op)
