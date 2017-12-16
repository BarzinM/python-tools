import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from os import environ
import numpy as np

from networks import fullyConnected, Convolutional, conv, flat, deconv
from tfmisc import getScopeParameters
from monitor import Figure

environ['CUDA_VISIBLE_DEVICES'] = ''

batch_size = 64
train_steps = 600000
latent_dim = 128
learning_rate = .0003

d_scale = .25
g_scale = .625

mnist = input_data.read_data_sets('MNIST')


# dataset = tf.data.Dataset.from_tensor_slices(
#     convert_to_tensor(CelebA(path).train_data_list, dtype=tf.string))
# dataset = dataset.map(lambda filename: tuple(tf.py_func(_read_by_function,
#                                                         [filename], [tf.double])), num_parallel_calls=16)
# dataset = dataset.repeat(100)
# dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
# iterator = tf.data.Iterator.from_structure(
#     dataset.output_types, dataset.output_shapes)
# training_init_op = iterator.make_initializer(dataset)

shape = (None, 28, 28)
input_data = tf.placeholder(tf.float32, shape)

flow_size = tf.shape(input_data)[0]


def encoder(flow):
    flow = conv('layer_0', flow, 32, 5, 2, tf.nn.relu, padding='SAME')
    print(flow.shape)
    flow = conv('layer_1', flow, 64, 5, 2, tf.nn.relu, padding='SAME')
    print(flow.shape)
    flow = flat(flow)
    mean = fullyConnected('mu', flow, latent_dim, None)
    sigma = fullyConnected('sigma', flow, latent_dim, None)

    return mean, sigma


def discriminator(flow):
    flow = conv('layer_0', flow, 32, 5, 2, tf.nn.relu, padding='SAME')
    flow = conv('layer_1', flow, 64, 5, 2, tf.nn.relu, padding='SAME')
    # flow = conv('layer_2', flow, 64, 5, 2, tf.nn.relu)
    # flow = conv('layer_3', flow, 64, 5, 2, None)
    middle_conv = flow
    flow = tf.nn.relu(flow)
    flow = flat(flow)
    flow = fullyConnected('layer_4', flow, 256, tf.nn.relu)
    flow = fullyConnected('output', flow, 1, None)

    return middle_conv, flow


def generator(flow):
    flow = fullyConnected('layer_0', flow, 7 * 7 * 64, tf.nn.relu)
    flow = tf.reshape(flow, [batch_size, 7, 7, 64])
    flow = tf.nn.relu(deconv('layer_1', flow, [batch_size, 14, 14, 32], 5, 2))
    print('gen', flow.shape)
    flow = tf.nn.relu(deconv('layer_2', flow, [batch_size, 28, 28, 32], 5, 2))
    print('gen', flow.shape)
    flow = deconv('output', flow, [batch_size, 28, 28, 1], 5, 1)
    print(flow.shape)
    return tf.nn.sigmoid(flow)


shaped_input = tf.reshape(input_data, (batch_size, 28, 28, 1))

with tf.variable_scope('encoder'):
    mean, sigma = encoder(shaped_input)

normal_sample = tf.random_normal(shape=(flow_size, latent_dim))
zp = tf.random_normal(shape=(flow_size, latent_dim))

z = mean + tf.multiply(tf.exp(.5 * sigma), normal_sample)

with tf.variable_scope('generator'):
    constructed = generator(z)

with tf.variable_scope('discriminator'):
    l_x_tilda, de_pro_tilde = discriminator(constructed)

with tf.variable_scope('generator', reuse=True):
    x_p = generator(zp)
    print('this', x_p.shape)

with tf.variable_scope('discriminator', reuse=True):
    lx, de_pro_logits = discriminator(shaped_input)
    _, g_pro_logits = discriminator(x_p)

kl_loss = -0.5 * tf.reduce_sum(1 + sigma - tf.square(mean) - tf.exp(sigma))

# disc loss
d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(g_pro_logits), logits=g_pro_logits)
d_fake_loss = tf.reduce_mean(d_fake_loss)
d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(de_pro_logits) - d_scale, logits=de_pro_logits)
d_real_loss = tf.reduce_mean(d_real_loss)
d_tilda_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(de_pro_tilde), logits=de_pro_tilde)
d_tilda_loss = tf.reduce_mean(d_tilda_loss)

d_loss = d_fake_loss + d_real_loss + d_tilda_loss


# gen loss
g_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(g_pro_logits) - g_scale, logits=g_pro_logits)
g_fake_loss = tf.reduce_mean(g_fake_loss)
g_tilda_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(de_pro_tilde) - g_scale, logits=de_pro_tilde)
g_tilda_loss = tf.reduce_mean(g_tilda_loss)

ll_loss = .5 * tf.square(l_x_tilda - lx) + -.5 * tf.log(2 * 3.1415)
ll_loss = tf.reduce_mean(ll_loss)

g_loss = g_fake_loss  # + g_tilda_loss - 1e-6 * ll_loss

# encode loss
encode_loss = tf.reduce_mean(
    kl_loss) / (batch_size * 784) - ll_loss  # TODO div(kl_loss, #)

disc_vars = getScopeParameters('discriminator')
gen_vars = getScopeParameters('generator')
encoder_vars = getScopeParameters('encoder')

global_step = tf.train.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(
    learning_rate, global_step, decay_steps=10000, decay_rate=.98)

optimizer = tf.train.AdamOptimizer(learning_rate)
disc_train = optimizer.minimize(d_loss, var_list=disc_vars)

optimizer = tf.train.AdamOptimizer(learning_rate)
gen_train = optimizer.minimize(g_loss, var_list=gen_vars)

optimizer = tf.train.AdamOptimizer(learning_rate*0)
enc_train = optimizer.minimize(encode_loss, var_list=encoder_vars)

init = tf.global_variables_initializer()

fig = Figure()

with tf.Session() as sess:
    sess.run(init)

    for step in range(10000):
        batch = mnist.train.next_batch(batch_size)[0]
        batch = np.reshape(batch, (-1, 28, 28))
        *losses, cons = sess.run([disc_train,
                                  gen_train,
                                  enc_train,
                                  d_loss,
                                  g_loss,
                                  encode_loss,
                                  constructed], {input_data: batch})[3:]

        print("G: %7.3f | D: %7.3f | E: %7.3f" % (*losses,))
        fig.imshow(np.concatenate(
            [batch[0], cons[0, :, :, 0]], axis=-1))
