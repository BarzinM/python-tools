import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from os import environ
import numpy as np

from networks import fullyConnected, Convolutional, conv, flat, deconv, lrelu, normalizeBatch
from tfmisc import getScopeParameters
from monitor import Figure

environ['CUDA_VISIBLE_DEVICES'] = ''

batch_size = 64
train_steps = 600000
latent_dim = 32
learning_rate = 0.0002

mnist = input_data.read_data_sets('MNIST')

shape = (None, 28, 28)
input_data = tf.placeholder(tf.float32, shape)

flow_size = tf.shape(input_data)[0]


def discriminator(flow):
    flow = conv('layer_0', flow, 32, 5, 2, None)
    flow = normalizeBatch(flow, True)
    flow = lrelu(flow)
    flow = conv('layer_1', flow, 64, 5, 2, None)
    flow = normalizeBatch(flow, True)
    flow = lrelu(flow)

    flow = flat(flow)
    flow = fullyConnected('layer_2', flow, 1024, None)
    flow = normalizeBatch(flow, True)
    flow = lrelu(flow)
    # flow = tf.nn.dropout(flow, .5)
    flow = fullyConnected('output', flow, 1, None)

    return flow


def generator(flow):
    flow = fullyConnected('layer_0', flow, 1024, None)
    flow = normalizeBatch(flow, True)
    flow = lrelu(flow)
    flow = fullyConnected('layer_1', flow, 7 * 7 * 64, None)
    flow = normalizeBatch(flow, True)
    flow = lrelu(flow)
    flow = tf.reshape(flow, [batch_size, 7, 7, 64])
    flow = deconv('layer_2', flow, [batch_size, 14, 14, 32], 5, 2)
    flow = normalizeBatch(flow, True)
    flow = lrelu(flow)
    flow = tf.nn.sigmoid(
        deconv('layer_3', flow, [batch_size, 28, 28, 1], 5, 2))
    return flow


shaped_input = tf.reshape(input_data, (batch_size, 28, 28, 1))

z_p = tf.random_normal(shape=(flow_size, latent_dim))

with tf.variable_scope('generator'):
    constructed = generator(z_p)

with tf.variable_scope('discriminator'):
    real_discrimination = discriminator(shaped_input)

with tf.variable_scope('discriminator', reuse=True):
    face_discrimination = discriminator(constructed)


# disc loss
d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(real_discrimination), logits=real_discrimination)
d_real_loss = tf.reduce_mean(d_real_loss)
d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(face_discrimination), logits=face_discrimination)
d_fake_loss = tf.reduce_mean(d_fake_loss)

d_loss = d_fake_loss + d_real_loss


# gen loss
g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(face_discrimination), logits=face_discrimination)
g_loss = tf.reduce_mean(g_loss)


disc_vars = getScopeParameters('discriminator')
gen_vars = getScopeParameters('generator')

global_step = tf.train.get_or_create_global_step()

optimizer = tf.train.AdamOptimizer(learning_rate)
disc_train = optimizer.minimize(d_loss, var_list=disc_vars)

optimizer = tf.train.AdamOptimizer(learning_rate)
gen_train = optimizer.minimize(g_loss, var_list=gen_vars)

init = tf.global_variables_initializer()

fig = Figure()

with tf.Session() as sess:
    sess.run(init)

    for step in range(train_steps):

        batch = mnist.train.next_batch(batch_size)[0]
        batch = np.reshape(batch, (-1, 28, 28))
        *losses, cons = sess.run([disc_train,
                                  gen_train,
                                  g_loss,
                                  d_loss,
                                  constructed], {input_data: batch})[2:]

        print("G: %7.3f | D: %7.3f" % (*losses,))
        fig.imshow(np.concatenate(
            [batch[0], cons[0, :, :, 0]], axis=-1), cmap='gray')
