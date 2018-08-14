import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from os import environ
import numpy as np

from networks import fullyConnected, Convolutional, conv, flat, deconv, normalizeBatch, lrelu
from tfmisc import getScopeParameters
from monitor import Figure

environ['CUDA_VISIBLE_DEVICES'] = ''

batch_size = 128
train_steps = 600000
latent_dim = 32
learning_rate = 0.0002

mnist = input_data.read_data_sets('MNIST')


shape = (None, 28, 28)
input_data = tf.placeholder(tf.float32, shape)

flow_size = tf.shape(input_data)[0]


def encoder(flow):
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

    mean = fullyConnected('mu', flow, latent_dim, None)
    sigma = fullyConnected('sigma', flow, latent_dim, None)

    return mean, sigma


def discriminator(flow):
    flow = conv('layer_0', flow, 32, 5, 2, None)
    flow = normalizeBatch(flow, True)
    flow = lrelu(flow)

    flow = conv('layer_1', flow, 64, 5, 2, None)
    flow = normalizeBatch(flow, True)
    flow = lrelu(flow)

    flow = flat(flow)

    flow = fullyConnected('layer_2', flow, 1024, None)
    layer_l = flow
    flow = normalizeBatch(flow, True)
    flow = lrelu(flow)
    # flow = tf.nn.dropout(flow, .5)

    flow = fullyConnected('output', flow, 1, None)

    return flow, layer_l


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


with tf.variable_scope('encoder'):
    mean, sigma = encoder(shaped_input)
    normal_sample = tf.random_normal(shape=(flow_size, latent_dim))
    sampled = mean + tf.multiply(tf.exp(.5 * sigma), normal_sample)


with tf.variable_scope('generator'):
    constructed = generator(sampled)

# Optional
# with tf.variable_scope('encoder', reuse=True):
#     recoded, _ = encoder(constructed)

with tf.variable_scope('generator', reuse=True):
    random_from_normal = tf.random_normal(shape=(flow_size, latent_dim))
    randomly_generated = generator(random_from_normal)


with tf.variable_scope('discriminator'):
    should_be_ones, l_dataset = discriminator(shaped_input)

with tf.variable_scope('discriminator', reuse=True):
    should_be_zeros, l_constructed = discriminator(constructed)

with tf.variable_scope('discriminator', reuse=True):
    from_normal, l_normal = discriminator(randomly_generated)

priori_loss = -0.5 * tf.reduce_sum(1 + sigma -
                                   tf.square(mean) - tf.exp(sigma), 1) / latent_dim
priori_loss = tf.reduce_mean(priori_loss)

# disc loss
d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(should_be_ones), logits=should_be_ones)
d_real_loss = tf.reduce_mean(d_real_loss)
d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(should_be_zeros), logits=should_be_zeros)
d_fake_loss = tf.reduce_mean(d_fake_loss)
d_random_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(from_normal), logits=from_normal)
d_random_loss = tf.reduce_mean(d_random_loss)

d_loss = d_fake_loss + d_real_loss + priori_loss


# gen loss
g_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(should_be_zeros), logits=should_be_zeros)
g_fake_loss = tf.reduce_mean(g_fake_loss)
g_random_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(from_normal), logits=from_normal)
g_random_loss = tf.reduce_mean(g_random_loss)

layer_l_loss = tf.reduce_mean(tf.square(l_constructed - l_dataset))

e_loss = priori_loss + layer_l_loss

# recoding_loss = tf.reduce_mean(tf.square(recoded) - sampled)

g_loss = layer_l_loss + (g_fake_loss + g_random_loss)  # + recoding_loss


disc_vars = getScopeParameters('discriminator')
gen_vars = getScopeParameters('generator')
encoder_vars = getScopeParameters('encoder')

global_step = tf.train.get_or_create_global_step()

optimizer = tf.train.AdamOptimizer(learning_rate)
disc_train = optimizer.minimize(d_loss, var_list=disc_vars)

optimizer = tf.train.AdamOptimizer(learning_rate)
gen_train = optimizer.minimize(g_loss, var_list=gen_vars)

optimizer = tf.train.AdamOptimizer(learning_rate * 0)
enc_train = optimizer.minimize(e_loss, var_list=encoder_vars)

init = tf.global_variables_initializer()

fig = Figure()

small_steps = 100
with tf.Session() as sess:
    sess.run(init)

    for step in range(train_steps):

        losses = np.zeros((3))
        for _ in range(small_steps):
            batch = mnist.train.next_batch(batch_size)[0]
            batch = np.reshape(batch, (-1, 28, 28))
            *l, cons = sess.run([enc_train,
                                 disc_train,
                                 gen_train,
                                 e_loss,
                                 g_loss,
                                 d_loss,
                                 constructed], {input_data: batch})[3:]
            losses += l

        print(step, "E: %7.3f | G: %7.3f | D: %7.3f" % (*(losses / small_steps),))
        fig.imshow(np.concatenate(
            [batch[0], cons[0, :, :, 0]], axis=-1), cmap='gray')
