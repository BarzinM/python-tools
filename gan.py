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

d_scale = .25
g_scale = .625

mnist = input_data.read_data_sets('MNIST')

shape = (None, 28, 28)
input_data = tf.placeholder(tf.float32, shape)

flow_size = tf.shape(input_data)[0]


# def encoder(flow):
#     flow = conv('layer_0', flow, 32, 5, 2, lrelu, padding='SAME')
#     print(flow.shape)
#     flow = conv('layer_1', flow, 64, 5, 2, lrelu, padding='SAME')
#     print(flow.shape)
#     flow = flat(flow)
#     mean = fullyConnected('mu', flow, latent_dim, None)
#     sigma = fullyConnected('sigma', flow, latent_dim, None)

#     return mean, sigma


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
    flow = tf.nn.dropout(flow, .5)
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

# with tf.variable_scope('encoder'):
#     mean, sigma = encoder(shaped_input)

# normal_sample = tf.random_normal(shape=(flow_size, latent_dim))
z_p = tf.random_normal(shape=(flow_size, latent_dim))

# z_x = mean + tf.multiply(tf.exp(.5 * sigma), normal_sample)

# with tf.variable_scope('generator'):
#     x_tilde = generator(z_x)

# with tf.variable_scope('discriminator'):
#     l_x_tilda, _ = discriminator(x_tilde)

with tf.variable_scope('generator'):
    x_p = generator(z_p)

constructed = x_p

with tf.variable_scope('discriminator'):
    d_x = discriminator(shaped_input)

with tf.variable_scope('discriminator', reuse=True):
    d_x_p = discriminator(x_p)


# sse_loss = tf.reduce_mean(tf.square(shaped_input - x_tilde))

# kl_loss = tf.reduce_sum(-.5 * tf.reduce_sum(1 + tf.clip_by_value(sigma, -10, 10) - tf.square(
# tf.clip_by_value(mean, -10, 10)) - tf.exp(tf.clip_by_value(sigma, -10,
# 10)), 1) / 28 / 28)

#-0.5 * tf.reduce_sum(1 + sigma - tf.square(mean) - tf.exp(sigma))

# disc loss
d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(d_x), logits=d_x)
d_real_loss = tf.reduce_mean(d_real_loss)
d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(d_x_p), logits=d_x_p)
d_fake_loss = tf.reduce_mean(d_fake_loss)
# d_tilda_loss = tf.nn.sigmoid_cross_entropy_with_logits(
#     labels=tf.zeros_like(de_pro_tilde), logits=de_pro_tilde)
# d_tilda_loss = tf.reduce_mean(d_tilda_loss)

d_loss = d_fake_loss + d_real_loss  # + d_tilda_loss

# eps = 1e-2
# d_loss = tf.reduce_mean(-tf.log(d_x + eps) - tf.log(1 - d_x_p + eps))
# g_loss = tf.reduce_mean(-tf.log(d_x_p + eps))

# d_loss = tf.reduce_mean(-1 * (tf.log(tf.clip_by_value(d_x, 1e-5, 1.)) +
# tf.log(tf.clip_by_value(1. - d_x_p, 1e-5, 1.))))

# gen loss
g_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(d_x_p), logits=d_x_p)
g_fake_loss = tf.reduce_mean(g_fake_loss)
# g_tilda_loss = tf.nn.sigmoid_cross_entropy_with_logits(
#     labels=tf.ones_like(de_pro_tilde) - g_scale, logits=de_pro_tilde)
# g_tilda_loss = tf.reduce_mean(g_tilda_loss)
# g_loss = g_fake_loss + g_tilda_loss - 1e-6 * ll_loss

g_loss = g_fake_loss

# ll_loss = tf.reduce_sum(tf.square(l_x - l_x_tilda)) / 28 / 28

# ll_loss = -.5 * tf.square(l_x_tilda - l_x) + -.5 * tf.log(2 * 3.1415)
# ll_loss = tf.reduce_mean(ll_loss)


# encode loss
# encode_loss = kl_loss / (batch_size * latent_dim) - ll_loss

# e_loss = kl_loss + ll_loss
# g_loss = (g_loss + ll_loss)


disc_vars = getScopeParameters('discriminator')
gen_vars = getScopeParameters('generator')
# encoder_vars = getScopeParameters('encoder')

global_step = tf.train.get_or_create_global_step()

optimizer = tf.train.AdamOptimizer(learning_rate)
disc_train = optimizer.minimize(d_loss, var_list=disc_vars)

optimizer = tf.train.AdamOptimizer(learning_rate)
gen_train = optimizer.minimize(g_loss, var_list=gen_vars)

# optimizer = tf.train.AdamOptimizer(learning_rate * 0)
# enc_train = optimizer.minimize(e_loss, var_list=encoder_vars)

init = tf.global_variables_initializer()

fig = Figure()

###########
# checked #

with tf.Session() as sess:
    sess.run(init)

    for step in range(10000):

        # for _ in range(5):
        #     batch = mnist.train.next_batch(batch_size)[0]
        #     batch = np.reshape(batch, (-1, 28, 28))
        #     sess.run(gen_train, {input_data: batch})

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
