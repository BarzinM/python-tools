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
            self.state = tf.placeholder(
                tf.float32, [None] + list(state_dim), "states")

        self.action_ph = tf.placeholder(tf.int32, [None], "actions")
        self.action_value_ph = tf.placeholder(
            tf.float32, [None], "action_values")
        self.memory = Memory(
            memory_size, (state_dim, np.float), (0, np.uint8), (0, float), (state_dim, np.float), (0, bool))
        print("Memory Bytes:", self.memory.nbytes() / (1024**3))

    def initialize(self, layer_dims, optimizer, learner_target_inputs=None):
        def _make(flow):
            for i, size in enumerate(layer_dims):
                flow = fullyConnected(
                    "layer%i" % i, flow, size, tf.nn.relu)

            return fullyConnected(
                "output_layer", flow, self.action_dim)

        learner_target_inputs = learner_target_inputs or [
            self.state, self.state]
        with tf.variable_scope('learner'):
            self.action_value = _make(learner_target_inputs[0])
        with tf.variable_scope('target'):
            self.target_action_value = _make(learner_target_inputs[1])

        row = tf.range(tf.shape(self.action_value)[0])
        indexes = tf.stack([row, self.action_ph], axis=1)

        updated = tf.Variable([], trainable=False, validate_shape=False)
        updated = tf.assign(updated, self.action_value, validate_shape=False)
        action_value = tf.scatter_nd_update(
            updated, indexes, self.action_value_ph)

        self._loss = tf.losses.huber_loss(
            self.action_value, action_value)

        self.policy_action = tf.argmax(self.action_value, axis=1)
        self.update_op = copyScopeVars('learner', 'target')

        self.train_op = optimizer.minimize(
            self._loss, var_list=getScopeParameters('learner'))

    def train(self, session, batch=None, discount=.97):
        states, actions, rewards, next_states, terminals = self.memory.sample(
            batch)
        next_state_value = session.run(
            self.target_action_value, {self.state: next_states})
        observed_value = rewards + (1. - terminals) * discount * \
            np.max(next_state_value, 1)

        return session.run([self._loss, self.train_op], {
            self.state: states,
            self.action_ph: actions,
            self.action_value_ph: observed_value})[0]

    def policy(self, session, state):
        return session.run(self.policy_action, {self.state: [state]})[0]

    def memorize(self, state, action, reward, next_state, terminal):
        self.memory.add(state, action, reward, next_state, terminal)

    def update(self, session):
        session.run(self.update_op)


class DQN2(object):

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
            memory_size, (state_dim, np.float), (0, np.uint8), (0, float), (state_dim, np.float), (0, bool))

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

        self.policy_action = tf.argmax(self.action_value, axis=1)
        self.update_op = copyScopeVars('learner', 'target')

        row = tf.range(0, tf.shape(self.action_value)[0])
        indexes = tf.stack([row, self.action_ph], axis=1)
        action_value = tf.gather_nd(self.action_value, indexes)

        # self.single_loss = tf.square(action_value - self.action_value_ph)
        # self._loss = tf.reduce_mean(self.single_loss)
        self._loss = tf.losses.huber_loss(self.action_value_ph, action_value)

        self.train_op = optimizer.minimize(self._loss)

    def train(self, session, batch=None, discount=.97):
        states, actions, rewards, next_states, terminals = self.memory.sample(
            batch)
        next_state_value = session.run(
            self.target_action_value, {self.state: next_states})
        observed_value = rewards + discount * \
            np.max(next_state_value, 1)
        observed_value[terminals] = rewards[terminals]

        _, l = session.run([self.train_op, self._loss], {
            self.state: states, self.action_ph: actions, self.action_value_ph: observed_value})
        return l

    def policy(self, session, state):
        return session.run(self.policy_action, {self.state: [state]})[0]

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
        self.state_ph = tf.placeholder(tf.float32, [None, state_dim], "states")
        self.action_ph = tf.placeholder(tf.int32, [None], "actions")
        self.reward_ph = tf.placeholder(tf.float32, [None], "rewards")
        self.next_state_ph = tf.placeholder(
            tf.float32, [None, state_dim], "states")
        self.terminal_ph = tf.placeholder(tf.float32, [None], "terminals")

        self.memory = Memory(
            memory_size, (state_dim, float), (0, int), (0, int), (state_dim, float), (0, bool))
        self.com_memory = Memory(
            memory_size, (state_dim, float), (0, int), (0, int), (state_dim, float), (0, bool))

        self.com_avg = RunningAverage(.001)
        self.com_avg(1.)
        self.mem_avg = RunningAverage(.001)

    def initialize(self, layer_dims=[100], optimizer=None):

        discount = .97

        def _make(flow):
            for i, size in enumerate(layer_dims):
                flow = fullyConnected(
                    "layer%i" % i, flow, size, tf.nn.relu)

            return fullyConnected(
                "output_layer", flow, self.action_dim, None)

        with tf.variable_scope('learner'):
            self.action_value = _make(self.state_ph)
        with tf.variable_scope('target'):
            self.target_action_value = _make(self.next_state_ph)

        self.update_op = copyScopeVars('learner', 'target')

        self.policy_action = tf.argmax(self.action_value, axis=1)

        self.batch_size = tf.shape(self.action_value)[0]
        row = tf.range(self.batch_size)
        indexes = tf.stack([row, self.action_ph], axis=1)

        next_q = tf.reduce_max(self.target_action_value, axis=1)
        target = self.reward_ph + discount * next_q * (1 - self.terminal_ph)
        target = tf.stop_gradient(target)

        prediction = tf.gather_nd(self.action_value, indexes)
        # self.q_loss = tf.losses.huber_loss(target, prediction)

        self.q_loss = tf.square(prediction - target)
        print('shape', self.q_loss.shape)
        self.get_grad = tf.gradients(self.q_loss, self.action_value)[0]
        print(self.get_grad.shape)
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

        try:
            states2, actions2, rewards2, next_states2, terminals2 = self.com_memory.sample(
                batch)

            states = np.concatenate((states, states2), 0)
            actions = np.concatenate((actions, actions2), 0)
            rewards = np.concatenate((rewards, rewards2), 0)
            next_states = np.concatenate((next_states, next_states2), 0)
            terminals = np.concatenate((terminals, terminals2), 0)
        except ValueError:
            pass

        dic = {self.state_ph: states,
               self.action_ph: actions,
               self.reward_ph: rewards,
               self.next_state_ph: next_states,
               self.terminal_ph: terminals}
        _, sl = session.run([self.train_op, self.q_loss], dic)

        if len(sl) > batch:
            self.com_avg(np.max(sl[batch:]))
        # self.com_avg.update(np.mean(sl))

        return np.mean(sl[:batch])

    def policy(self, session, state):
        return session.run(self.action_value, {self.state_ph: [state]})[0]

    def memorize(self, session, state, action, reward, next_state, terminal):
        # if not self.memory.isFull():
        #     self.memory.add(state, action, reward, next_state, terminal)
        #     return
        # next_state_value = session.run(
        #     self.target_action_value, {self.state_ph: [next_state]})
        # observed_value = reward + .97 * \
        #     np.max(next_state_value, 1, keepdims=True)
        # if terminal:
        #     observed_value[0, 0] = reward / .03
        dic = {self.state_ph: state[None, :],
               self.action_ph: [action],
               self.reward_ph: [reward],
               self.next_state_ph: [next_state],
               self.terminal_ph: [terminal]}

        l = session.run(self.q_loss, dic)[0]

        if l > self.com_avg():
            self.com_memory.add(state, action, reward, next_state, terminal)
        else:
            self.memory.add(state, action, reward, next_state, terminal)
            # print("skipping")

    def update(self, session):
        session.run(self.update_op)
