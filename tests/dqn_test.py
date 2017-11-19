import tensorflow as tf
import gym
import numpy as np
from agents.dqn import DQN2
from os import environ
from time import time

environ['CUDA_VISIBLE_DEVICES'] = ''


session = tf.InteractiveSession()
agent = DQN2(4, 2, 2000)
agent.initialize([200], tf.train.AdamOptimizer(.01))
env = gym.make('CartPole-v1')


def main(count, verb=False):
    epsilon = [.5]

    def runEpisode(env, session, train=True):
        state = env.reset()
        loss = 0.
        i = 0
        while True:
            if np.random.uniform() < epsilon[0]:
                action = np.random.randint(2)
            else:
                action = agent.policy(session, state)

            next_state, reward, done, info = env.step(action)

            if done:
                reward = -1.
            else:
                reward = .01

            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            loss += agent.train(session, 32)

            i += 1

            if done or i == 200:
                break

        epsilon[0] = epsilon[0] * .99
        agent.update(session)
        return i, loss

    session.run(tf.global_variables_initializer())
    avg = 0
    avg_l = 0
    start = time()
    for i in range(count):
        leng, l = runEpisode(env, session)
        if verb and not i % 50:
            print(i, leng)
        avg += leng
        avg_l += l

    return avg / count, avg_l / avg, (time() - start) / avg


count = 50
length = 1000
res = []
times = 0.
for i in range(count):
    start = time()
    step, loss, t = main(length, False)
    times += t
    res.append(step)
    print(i + 1, np.mean(res), np.std(res), loss, times / (i + 1))
print("END", np.mean(res), np.std(res))
print("times:", np.mean(times), np.std(times))
session.close()
