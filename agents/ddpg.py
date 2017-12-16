import tensorflow as tf
import gym
import numpy as np
from os import environ

from networks import fullyConnected
from tfmisc import copyScopeVars, getScopeParameters
from memory import ContinuousMemory
from base_agent import BaseAgent
from monitor import Monitor

environ['CUDA_VISIBLE_DEVICES'] = ''


environment_name = "Pendulum-v0"
ACTOR_LEARNING_RATE = .0001
CRITIC_LEARNING_RATE = .001
TAU = .001
BUFFER_SIZE = 10000
TRAINING_EPISODES = 250
MAX_STEPS = 200
BATCH_SIZE = 64
GAMMA = .99

RANDOM_SEED = 1234
SUMMARY_PATH = './summary/new-ddpg'

env = gym.envs.make(environment_name)
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


tf.reset_default_graph()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bounds = env.action_space.high
print("%i states, %i actions, %.3f action bound" %
      (state_dim, action_dim, action_bounds))
assert env.action_space.high == -env.action_space.low


class DDPG(object):

    def __init__(self, state_dim, action_dim, actor_dimensions, critic_dimensions):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.train_count = 0

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build(actor_dimensions, critic_dimensions)
            self.init = tf.global_variables_initializer()

        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)

    def _build(self, actor_dimensions, critic_dimensions):
        self.global_step = tf.train.get_or_create_global_step()

        self.reward_ph = tf.placeholder(tf.float32, (None,), 'rewards')
        self.terminal_ph = tf.placeholder(tf.bool, (None,), 'terminals')

        self.state_ph = tf.placeholder(
            tf.float32, (None, self.state_dim), 'states')
        self.next_state_ph = tf.placeholder(
            tf.float32, (None, self.state_dim), 'next_states')

        self.batch_size = tf.shape(self.state_ph)[0]

        def makeActor(flow):
            for i, dim in enumerate(actor_dimensions):
                flow = fullyConnected("layer_%i" % i, flow, dim, tf.nn.relu)
            flow = fullyConnected("policy", flow, self.action_dim, tf.nn.tanh)
            flow = flow * action_bounds
            return flow

        def makeCritic(flow_1, flow_2):
            flow = [flow_1, flow_2]
            for i, dim in enumerate(critic_dimensions):
                flow = fullyConnected("layer_%i" % i, flow, dim, tf.nn.relu)
            flow = fullyConnected("value", flow, 1, None)
            flow = tf.reshape(flow, (self.batch_size,))
            return flow

        with tf.variable_scope("learner/actor"):
            self.actor_policy = makeActor(self.state_ph)
        with tf.variable_scope("learner/critic"):
            self.value = makeCritic(self.state_ph, self.actor_policy)
        with tf.variable_scope("target/actor"):
            next_policy = makeActor(self.next_state_ph)
        with tf.variable_scope("target/critic"):
            next_value = makeCritic(self.next_state_ph, next_policy)

        self.update_op = copyScopeVars("learner", "target", TAU)
        actor_parameters = getScopeParameters("learner/actor")
        critic_parameters = getScopeParameters("learner/critic")

        actor_optimizer = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE)
        critic_optimzier = tf.train.AdamOptimizer(CRITIC_LEARNING_RATE)

        grads = tf.gradients(self.value, self.actor_policy)[0]
        grads = tf.div(grads, tf.to_float(self.batch_size))
        actor_grads = tf.gradients(
            self.actor_policy, actor_parameters, -grads)
        gnv = zip(actor_grads, actor_parameters)
        self.train_actor_op = actor_optimizer.apply_gradients(gnv)

        terminal = tf.to_float(self.terminal_ph)
        target_value = self.reward_ph + GAMMA * next_value * (1. - terminal)
        target_value = tf.stop_gradient(target_value)

        critic_loss = tf.losses.huber_loss(target_value, self.value)
        self.debug = critic_loss
        self.train_critic_op = critic_optimzier.minimize(
            critic_loss, var_list=critic_parameters)

    def policy(self, state):
        return self.session.run(self.actor_policy, {self.state_ph: state[None, :]})[0]

    def train(self, state, action, reward, next_state, terminal):
        d = self.session.run(
            [self.debug,
             self.train_critic_op],
            {self.state_ph: state,
             self.actor_policy: action,
             self.reward_ph: reward,
             self.next_state_ph: next_state,
             self.terminal_ph: terminal})[:-1]

        self.train_count, q = self.session.run(
            [self.global_step,
             self.value,
             self.update_op,
             self.train_actor_op],
            {self.state_ph: state})[:-2]

        return q

mon = Monitor(sub='separate_actor_critic')

print("action_dim", action_dim, state_dim)
memory = ContinuousMemory(BUFFER_SIZE, (state_dim, float), (action_dim, float))
agent = DDPG(state_dim, action_dim, [400, 300], [400, 300])

for episode in range(TRAINING_EPISODES):
    state = env.reset()
    episode_reward = 0.
    average_max_q = 0.
    memory.add(state)

    for step in range(MAX_STEPS):
        action = agent.policy(state)
        new_state, reward, terminal, _ = env.step(action)
        memory.add(new_state, action, reward, terminal)
        episode_reward += reward

        if terminal:
            break

        state, action, reward, terminal = memory.sample(BATCH_SIZE)
        q = agent.train(state[0], action, reward, state[-1], terminal)
        average_max_q += np.amax(q)

        state = new_state

    print(episode, step, episode_reward, average_max_q / step)
    mon.update({'episode_reward': episode_reward,
                'q': average_max_q / step}, episode)
