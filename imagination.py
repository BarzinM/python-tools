import tensorflow as tf
from approximators import fullyConnected


def imaginationNN(inputs, state_dim):
    state = fullyConnected("state_prediction", inputs, state_dim)
    reward = fullyConnected("reward_prediction", inputs, 1)
    termination = fullyConnected(
        "termination_prediction", inputs, 1, activation=tf.nn.sigmoid)
    return state, reward, termination
