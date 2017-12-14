from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from os import environ
import numpy as np
from time import sleep

from networks import fullyConnected, latent
from monitor import Figure

environ['CUDA_VISIBLE_DEVICES'] = ''


mnist = input_data.read_data_sets('MNIST')

input_dim = 784
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
lam = 0

input_data = tf.placeholder("float", shape=[None, input_dim])

flow = fullyConnected('hidden', input_data, hidden_encoder_dim, tf.nn.relu)

z, kl_loss = latent(flow, latent_dim)

flow = fullyConnected(
    'hidden_decoder', z, hidden_decoder_dim, tf.nn.relu)
# flow = fullyConnected('hidden_2', flow, 600, tf.nn.relu)
x_hat = fullyConnected('output', flow, input_dim, None)

generated = tf.sigmoid(x_hat)

# reconstruction_loss = tf.reduce_sum(tf.square(generated - input_data), 1)
reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=x_hat, labels=input_data), reduction_indices=1)
loss = tf.reduce_mean(reconstruction_loss + kl_loss)

train_step = tf.train.AdamOptimizer(0.01).minimize(loss)


n_steps = int(1e6)
batch_size = 256

fig = Figure()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1, n_steps):
        batch = mnist.train.next_batch(batch_size)[0]
        feed_dict = {input_data: batch}
        _, cur_loss, output = sess.run(
            [train_step, loss, generated], feed_dict=feed_dict)
        print(step, cur_loss)
        original = np.reshape(batch[0], (28, 28))
        reconstructed = np.reshape(output[0], (28, 28))
        sidebyside = np.concatenate([original, reconstructed], axis=1)
        fig.imshow(sidebyside)
        if not step % 50:
            sleep(3)
