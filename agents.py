import numpy as np
import tensorflow as tf
from approximators import fullyConnected, copyScopeVars, getScopeParameters, entropyLoss
from memory import Memory, GeneralMemory


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
        self.input = tf.placeholder(tf.float32, [None, state_dim])
        self.action_ph = tf.placeholder(tf.float32, [None, action_dim])
        self.memory = Memory(memory_size, state_dim, 0)

    def initialize(self, layer_dims=[100], optimizer=None):
        flow = self.input
        for i, size in enumerate(layer_dims):
            flow = fullyConnected("layer%i" % i, flow, size, tf.nn.relu)

        self.action_value = fullyConnected(
            "output_layer", flow, self.action_dim)
        self._loss = tf.reduce_mean(
            tf.square(self.action_value - self.action_ph))

        self.optimizer = optimizer or tf.train.AdamOptimizer(.01)
        self.train_op = self.optimizer.minimize(self._loss)

    def train(self, session, batch=None, discount=.97):
        states, actions, rewards, next_states, terminals = self.memory.sample(
            batch)
        action_value = session.run(
            self.action_value, {self.input: states})
        next_state_value = session.run(
            self.action_value, {self.input: next_states})
        observed_value = rewards + discount * \
            np.max(next_state_value, 1, keepdims=True)
        observed_value[terminals] = rewards[terminals] / (1 - discount)
        action_value[np.arange(batch), actions] = observed_value[:, 0]

        _, l = session.run([self.train_op, self._loss], {
            self.input: states, self.action_ph: action_value})
        return l

    def policy(self, session, state):
        return session.run(self.action_value, {self.input: [state]})[0]

    def memorize(self, state, action, reward, next_state, terminal):
        self.memory.add(state, action, reward, next_state, terminal)


class DuelingDQN(DQN):
    def initialize(self, layer_dims=[100], optimizer=None):
        flow = self.input
        for i, size in enumerate(layer_dims):
            flow = fullyConnected("layer%i" % i, flow, size, tf.nn.relu)

        value = fullyConnected("value", flow, 1)
        advantage = fullyConnected("advantage", flow, self.action_dim)
        self.action_value = value + advantage - \
            tf.reduce_mean(advantage, axis=1, keep_dims=True)
        self._loss = tf.reduce_mean(
            tf.square(self.action_value - self.action_ph))

        self.optimizer = optimizer or tf.train.AdamOptimizer(.01)
        self.train_op = self.optimizer.minimize(self._loss)


class DoubleDQN(DQN):
    def initialize(self, layer_dims=[100], optimizer=None):
        with tf.variable_scope('learner'):
            flow = self.input
            for i, size in enumerate(layer_dims):
                flow = fullyConnected("layer%i" % i, flow, size, tf.nn.relu)

            self.action_value = fullyConnected(
                "output_layer", flow, self.action_dim)

        with tf.variable_scope('target'):
            flow = self.input
            for i, size in enumerate(layer_dims):
                flow = fullyConnected("layer%i" % i, flow, size, tf.nn.relu)

            self.target_action_value = fullyConnected(
                "output_layer", flow, self.action_dim)

        self.update_op = copyScopeVars('learner', 'target')

        self._loss = tf.reduce_mean(
            tf.square(self.action_value - self.action_ph))

        self.optimizer = optimizer or tf.train.AdamOptimizer(.01)
        self.train_op = self.optimizer.minimize(self._loss)

    def train(self, session, batch=None, discount=.97):
        states, actions, rewards, next_states, terminals = self.memory.sample(
            batch)
        action_value = session.run(
            self.action_value, {self.input: states})
        next_action = np.argmax(action_value,axis=1)
        next_state_value = session.run(
            self.target_action_value, {self.input: next_states})
        observed_value = rewards + np.expand_dims(discount * next_state_value[np.arange(batch),next_action],-1)
        # print(observed_value.shape,np.expand_dims(discount * next_state_value[np.arange(batch),next_action],-1).shape,rewards.shape)
        # raise
        observed_value[terminals] = rewards[terminals] / (1 - discount)
        action_value[np.arange(batch), actions] = observed_value[:, 0]

        _, l = session.run([self.train_op, self._loss], {
            self.input: states, self.action_ph: action_value})
        return l

    def update(self, session):
        session.run(self.update_op)


class DoubleDuelingDQN(DoubleDQN):
    def initialize(self, layer_dims=[100], optimizer=None):
        with tf.variable_scope('learner'):
            flow = self.input
            for i, size in enumerate(layer_dims):
                flow = fullyConnected("layer%i" % i, flow, size, tf.nn.relu)

            value = fullyConnected("output_layer", flow, 1)
            advantage = fullyConnected("advantage", flow, self.action_dim)
            self.action_value = value + advantage - \
                tf.reduce_mean(advantage, axis=1, keep_dims=True)

        with tf.variable_scope('target'):
            flow = self.input
            for i, size in enumerate(layer_dims):
                flow = fullyConnected("layer%i" % i, flow, size, tf.nn.relu)

            target_value = fullyConnected("output_layer", flow, 1)
            target_advantage = fullyConnected(
                "advantage", flow, self.action_dim)
            self.target_action_value = target_value + target_advantage - \
                tf.reduce_mean(target_advantage, axis=1, keep_dims=True)

        self.update_op = copyScopeVars('learner', 'target')

        self._loss = tf.reduce_mean(
            tf.square(self.action_value - self.action_ph))

        self.optimizer = optimizer or tf.train.AdamOptimizer(.01)
        self.train_op = self.optimizer.minimize(self._loss)


class HighDimDQN(DQN):
    def __init__(self, reading_ph, pose_ph, input_tensor, action_dim, memory_size):
        self.reading_ph = reading_ph
        self.pose_ph = pose_ph
        self.input = input_tensor

        self.action_dim = action_dim
        self.action_ph = tf.placeholder(tf.float32, [None, action_dim])

        reading_dim = reading_ph.get_shape()[-1]
        pose_dim = pose_ph.get_shape()[-1]
        self.memory = GeneralMemory(
            memory_size, [reading_dim, pose_dim, 0, 1, reading_dim, pose_dim, 0])

    def train(self, session, batch=None, discount=.97):
        reading, pose, actions, rewards, next_reading, next_pose, terminals = self.memory.sample(
            batch)
        action_value = session.run(
            self.action_value, {self.reading_ph: reading, self.pose_ph: pose})
        next_state_value = session.run(
            self.action_value, {self.reading_ph: next_reading, self.pose_ph: next_pose})
        observed_value = rewards + discount * \
            np.max(next_state_value, 1, keepdims=True)
        observed_value[terminals] = rewards[terminals] / (1 - discount)
        action_value[np.arange(batch), actions] = observed_value[:, 0]

        _, l = session.run([self.train_op, self._loss], {
            self.reading_ph: reading, self.pose_ph: pose, self.action_ph: action_value})
        return l

    def policy(self, session, reading, pose):
        return session.run(self.action_value, {self.reading_ph: [reading], self.pose_ph: [pose]})[0]

    def memorize(self, reading, pose, action, reward, next_reading, next_pose, terminal):
        self.memory.add([reading, pose, action, reward,
                         next_reading, next_pose, terminal])


class HighDimDuelingDQN(HighDimDQN, DuelingDQN):
    pass


class HighDimDoubleDQN(HighDimDQN, DoubleDQN):
    def train(self, session, batch=None, discount=.97):
        reading, pose, actions, rewards, next_reading, next_pose, terminals = self.memory.sample(
            batch)
        action_value = session.run(
            self.action_value, {self.reading_ph: reading, self.pose_ph: pose})
        next_state_value = session.run(
            self.target_action_value, {self.reading_ph: next_reading, self.pose_ph: next_pose})
        observed_value = rewards + discount * \
            np.max(next_state_value, 1, keepdims=True)
        observed_value[terminals] = rewards[terminals] / (1 - discount)
        action_value[np.arange(batch), actions] = observed_value[:, 0]

        _, l = session.run([self.train_op, self._loss], {
            self.reading_ph: reading, self.pose_ph: pose, self.action_ph: action_value})
        return l


class HighDimDoubleDuelingDQN(HighDimDoubleDQN, DoubleDuelingDQN):
    pass


initializer = tf.random_uniform_initializer(minval=-0.005, maxval=0.005)


class DDPG(object):
    def __init__(self, state_dim, action_dim, memory_size, tau=.01):
        self.action_dim = action_dim
        self.tau = tau
        self.state = tf.placeholder(tf.float32, (None, state_dim))
        self.action_ph = tf.placeholder(tf.float32, (None, action_dim))
        self.grad_ph = tf.placeholder(tf.float32, (None, action_dim))
        self.value_ph = tf.placeholder(tf.float32, (None, 1))
        self.memory = Memory(memory_size, state_dim, 0)
        self.update = []

    def initializeActor(self, layers=[400, 300], bounds=None, optimizer=None):
        optimizer = optimizer or tf.train.AdamOptimizer(.001)

        def _make(bounds):
            net = self.state
            for layer in layers:
                net = fullyConnected("layer_%i" %
                                     layer, net, layer, tf.nn.relu, initializer)
            actions = fullyConnected(
                "actor_output", net, self.action_dim, tf.nn.tanh, initializer)
            if bounds is not None:
                actions = tf.multiply(actions, tf.constant(bounds))
            return actions

        with tf.variable_scope("learner/actor"):
            self.actor = _make(bounds)
        with tf.variable_scope("target/actor"):
            self.target_actor = _make(bounds)

        self.update.append(copyScopeVars(
            "learner/actor", "target/actor", self.tau))

        parameters = getScopeParameters("learner/actor")
        grads = tf.gradients(self.actor, parameters, -self.grad_ph)

        grads_and_vars = zip(grads, parameters)
        self.train_actor = optimizer.apply_gradients(grads_and_vars)

    def initializeCritic(self, layers=[400, 300], optimizer=None):
        assert len(layers) == 2, "THIS SHOULD BE FIXED!!!"
        optimizer = optimizer or tf.train.AdamOptimizer(.1)

        def _make():
            net = fullyConnected("state_layer", self.state,
                                 layers[0], tf.nn.relu, initializer)
            net = fullyConnected(
                "mixed_layer_1", [net, self.action_ph], layers[1], tf.nn.relu, initializer)
            return fullyConnected("critic_output", net, 1, initializer=initializer)

        with tf.variable_scope("learner/critic"):
            self.critic = _make()
        with tf.variable_scope("target/critic"):
            self.target_critic = _make()

        self.update.append(copyScopeVars(
            "learner/critic", "target/critic", self.tau))

        self._loss = tf.nn.l2_loss(self.value_ph - self.critic)
        self.train_critic = optimizer.minimize(self._loss)
        self.get_grads = tf.div(tf.gradients(self.critic, self.action_ph), 16)

    def train(self, session, batch_size, discount=.97):
        states, actions, rewards, next_states, terminals = self.memory.sample(
            batch_size)
        next_action = session.run(
            self.target_actor, {self.state: next_states})
        next_action_value = session.run(
            self.target_critic, {self.state: next_states, self.action_ph: next_action})
        observed_value = rewards + discount * next_action_value
        observed_value[terminals] = rewards[terminals] / (1 - discount)

        l, _ = session.run([self._loss, self.train_critic], {
            self.value_ph: observed_value, self.state: states, self.action_ph: actions})

        probable_action = session.run(self.actor, {self.state: states})
        grads = session.run(
            self.get_grads, {self.state: states, self.action_ph: probable_action})[0]
        session.run(self.train_actor, {
                    self.state: states, self.grad_ph: grads})
        session.run(self.update)
        return l

    def policy(self, session, state):
        return session.run(self.actor, {self.state: [state]})[0]

    def memorize(self, state, action, reward, next_state, terminal):
        self.memory.add(state, action, reward, next_state, terminal)


class ActorCritic(object):
    def __init__(self, state_dim, action_dim, memory_size, shared_net=[]):
        self.action_dim = action_dim
        self.state = tf.placeholder(tf.float32, (None, state_dim))
        self.shared = self.state
        with tf.variable_scope("shared"):
            for i, layer in enumerate(shared_net):
                self.shared = fullyConnected(
                    "layer_%i" % i, self.shared, layer, tf.nn.relu)
        self.action_ph = tf.placeholder(tf.float32, (None, action_dim))
        self.grad_ph = tf.placeholder(tf.float32, (None, 1))
        self.value_ph = tf.placeholder(tf.float32, (None, 1))
        self.memory = Memory(memory_size, state_dim, 0)

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

        # print(tf.expand_dims(tf.range(0, tf.shape(self.actor)[
        #       0]), axis=1).get_shape(), self.action_ph.get_shape())
        # indexes = tf.concat(
        #     [tf.expand_dims(tf.range(0, tf.shape(self.actor)[0]), axis=1), tf.expand_dims(self.action_ph,axis=1)], axis=1)
        # print("index", indexes.get_shape())
        # action_probibility = tf.gather_nd(self.actor, indexes)
        # action_probibility = tf.slice(self.actor,self.action_ph,[[1]])

        action_probibility = tf.reduce_sum(
            tf.multiply(self.action_ph, self.actor), axis=[1], keep_dims=True)
        self.single_loss = tf.log(action_probibility + 1e-8) * self.grad_ph
        self.actor_loss = -tf.reduce_sum(self.single_loss)
        if entropy != 0.0:
            self.actor_loss -= entropyLoss(self.actor) * entropy

        self.train_actor = optimizer.minimize(self.actor_loss)

    def initializeCritic(self, layers=[400, 300], optimizer=None):
        optimizer = optimizer or tf.train.AdamOptimizer(.001)

        def _make():
            net = self.shared
            for i, layer in enumerate(layers):
                net = fullyConnected("layer_%i" %
                                     i, net, layer, tf.nn.relu, .1)
            return fullyConnected("critic_output", net, 1, initializer=.0001)

        with tf.variable_scope("critic"):
            self.critic = _make()

        self.critic_loss = tf.reduce_mean(
            tf.square(self.value_ph - self.critic))
        self.train_critic = optimizer.minimize(self.critic_loss)

    def train(self, session, batch_size, discount=.97):
        states, actions, rewards, next_states, terminals = self.memory.sample(
            batch_size)
        next_state_value = session.run(
            self.critic, {self.state: next_states})
        state_value = session.run(
            self.critic, {self.state: states})
        observed_value = rewards + discount * next_state_value
        observed_value[terminals] = rewards[terminals] / (1 - discount)
        actions_onehot = np.zeros((batch_size, self.action_dim), float)
        actions_onehot[np.arange(batch_size), actions] = 1
        # print(actions,actions_onehot)
        l, _ = session.run([self.critic_loss, self.train_critic], {
                           self.value_ph: observed_value, self.state: states})
        _, actor_l = session.run([self.train_actor, self.actor_loss], {
            self.state: states, self.action_ph: actions_onehot, self.grad_ph: observed_value - state_value})

        return l

    def policy(self, session, state):
        a = session.run(self.actor, {self.state: [state]})[0]
        return a

    def memorize(self, state, action, reward, next_state, terminal):
        self.memory.add(state, action, reward, next_state, terminal)
