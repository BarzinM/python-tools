import numpy as np
import tensorflow as tf
from approximators import fullyConnected, copyScopeVars, getScopeParameters, entropyLoss
from memory import Memory, GeneralMemory
from time import time
from a3c import A3C
from dqn import *


def choice(probabilities):
    return np.random.choice(len(probabilities), p=probabilities)


def epsGreedy(probabilities, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(probabilities))
    else:
        return np.argmax(probabilities)


def greedy(probabilities):
    return np.argmax(probabilities)



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
                                     layer, net, layer, tf.nn.relu, 0.005)
            actions = fullyConnected(
                "actor_output", net, self.action_dim, tf.nn.tanh, 0.005)
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
                                 layers[0], tf.nn.relu, 0.005)
            net = fullyConnected(
                "mixed_layer_1", [net, self.action_ph], layers[1], tf.nn.relu, 0.005)
            return fullyConnected("critic_output", net, 1, initializer=0.005)

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

        action_probibility = tf.reduce_sum(
            tf.multiply(self.action_ph, self.actor), axis=[1], keep_dims=True)
        self.single_loss = tf.log(action_probibility + 1e-8) * self.grad_ph
        self.actor_loss = -tf.reduce_sum(self.single_loss)
        if entropy != 0.0:
            self.actor_loss -= entropyLoss(self.actor) * entropy

        self.train_actor = optimizer.minimize(self.actor_loss)

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


class DiscreteActorCritic(ActorCritic):
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


class ExpActorCritic(ActorCritic):
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

        self.memory = GeneralMemory(
            memory_size, state_dim, 0, 1, state_dim, -1)
        self.memory_average = RunningAverage(.01)
        self.com_memory = GeneralMemory(
            memory_size, state_dim, 0, 1, state_dim, -1)
        self.com_average = RunningAverage(.01)

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

        self.critic_single_loss = tf.square(self.value_ph - self.critic)
        self.critic_loss = tf.reduce_mean(self.critic_single_loss)

        self.train_critic = optimizer.minimize(self.critic_loss)

    def train(self, session, batch_size, discount=.97):
        try:
            states, actions, rewards, next_states, terminals = self.memory.sample(
                batch_size)
        except ValueError:
            return 0
        next_state_value = session.run(
            self.critic, {self.state: next_states})
        state_value = session.run(
            self.critic, {self.state: states})
        observed_value = rewards + discount * next_state_value
        observed_value[terminals] = rewards[terminals] / (1 - discount)

        single_loss, loss, _ = session.run([self.critic_single_loss, self.critic_loss, self.train_critic], {
            self.value_ph: observed_value, self.state: states})

        self.memory_average.update(loss)
        # tops = np.squeeze(single_loss > self.com_average.avg)
        # self.com_memory.addBatch(states[tops], actions[tops], rewards[
        #                          tops], next_states[tops], terminals[tops])

        _, actor_l = session.run([self.train_actor, self.actor_loss], {
            self.state: states, self.action_ph: actions, self.grad_ph: observed_value - state_value})
        return loss  # , 0  # sum(tops)

    def train2(self, session, batch_size, discount=.97):
        states, actions, rewards, next_states, terminals = self.com_memory.sample(
            batch_size)
        next_state_value = session.run(
            self.critic, {self.state: next_states})
        # state_value = session.run(
        #     self.critic, {self.state: states})
        observed_value = rewards + discount * next_state_value
        observed_value[terminals] = rewards[terminals] / (1 - discount)

        com_single_loss, com_loss, _ = session.run([self.critic_single_loss, self.critic_loss, self.train_critic], {
            self.value_ph: observed_value, self.state: states})
        self.com_average.update(com_loss)
        return com_loss

    def memorize(self, state, action, reward, next_state, terminal, session):
        next_state_value = session.run(
            self.critic, {self.state: [next_state]})[0]
        state_value = session.run(
            self.critic, {self.state: [state]})[0]
        observed_value = reward + .97 * next_state_value
        loss = np.square(observed_value - state_value)
        # print(loss, next_state_value, state_value, observed_value)
        # raise
        if loss > self.com_average.avg:
            self.com_memory.add(state, action, reward, next_state, terminal)
            return 1
        elif True:
            self.memory.add(state, action, reward, next_state, terminal)
        return 0


class ExpActorCritic2(ActorCritic):
    def __init__(self, state_dim, action_dim, memory_size, shared_net=[]):
        self.action_dim = action_dim
        self.state = tf.placeholder(
            tf.float32, (None, state_dim), name='state')

        self.shared = self.state
        with tf.variable_scope("shared"):
            for i, layer in enumerate(shared_net):
                self.shared = fullyConnected(
                    "layer_%i" % i, self.shared, layer, tf.nn.relu)

        self.action_ph = tf.placeholder(tf.int32, (None))
        self.grad_ph = tf.placeholder(tf.float32, (None, 1))
        self.value_ph = tf.placeholder(tf.float32, (None, 1))

        self.memory = GeneralMemory(
            memory_size, state_dim, 0, 1, state_dim, -1)
        self.com_memory = GeneralMemory(
            memory_size, state_dim, 0, 1, state_dim, -1)
        self.com_average = RunningAverage(.001)

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

        self.critic_single_loss = tf.square(self.value_ph - self.critic)
        self.critic_loss = tf.reduce_mean(self.critic_single_loss)
        self.get_grad = tf.gradients(self.critic_single_loss, self.critic)
        self.clipped_grad_ph = tf.placeholder(tf.float32, [None, 1])

        variables = getScopeParameters("critic")
        grads = tf.gradients(self.critic, variables, self.clipped_grad_ph)
        clipped = zip(grads, variables)

        self.train_critic = optimizer.apply_gradients(clipped)

    def _train(self, memory, session, batch_size, discount=.97):
        try:
            states, actions, rewards, next_states, terminals = memory.sample(
                batch_size)
        except ValueError:
            states, actions, rewards, next_states, terminals = self.memory.sample(
                batch_size)
        next_state_value = session.run(
            self.critic, {self.state: next_states})
        state_value = session.run(
            self.critic, {self.state: states})
        observed_value = rewards + discount * next_state_value
        observed_value[terminals] = rewards[terminals] / (1 - discount)

        grad, single_loss, loss = session.run([self.get_grad, self.critic_single_loss, self.critic_loss], {
            self.value_ph: observed_value, self.state: states})

        self.com_average.update(loss)
        return states, actions, observed_value - state_value, grad[0]

    def train(self, session, batch_size, discount=.97):
        s, a, v, g = self._train(self.memory, session, batch_size, discount)
        cs, ca, cv, cg = self._train(
            self.com_memory, session, batch_size, discount)

        session.run(self.train_critic, {
                    self.state: np.concatenate([s, cs]), self.clipped_grad_ph: np.concatenate([g, cg])})

        _, actor_l = session.run([self.train_actor, self.actor_loss], {
            self.state: np.concatenate([s, cs]), self.action_ph: np.concatenate([a, ca]), self.grad_ph: np.concatenate([v, cv])})

        return 0

    def grad(self):
        return np.min(np.abs(self.g))

    def memorize(self, state, action, reward, next_state, terminal, session):
        next_state_value = session.run(
            self.critic, {self.state: [next_state]})[0][0]
        state_value = session.run(
            self.critic, {self.state: [state]})[0][0]
        observed_value = reward + .97 * next_state_value
        loss = np.square(observed_value - state_value)

        if loss < self.com_average.avg or self.memory.count() == 0:
            self.memory.add(state, action, reward, next_state, terminal)
            return 0, loss
        else:
            self.com_memory.add(state, action, reward, next_state, terminal)
            return 1, loss

    def sleep(self, session, duration):
        start_time = time()
        while self.grad() > .001:
            self.train(session, 8)
            if time() - start_time > duration:
                print("time up")
                return
