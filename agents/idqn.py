import numpy as np
import tensorflow as tf
from approximators import fullyConnected, copyScopeVars, getScopeParameters
from memory import GeneralMemory
from agents.dqn import DQN
from matplotlib.pyplot import figure, imshow, axis, show, plot
from matplotlib.image import imread


class ImaginationDQN(DQN):
    def __init__(self, state_dim, action_dim, memory_size):
        super(ImaginationDQN, self).__init__(
            state_dim, action_dim, memory_size)
        self.state_dim = state_dim
        self.reward_ph = tf.placeholder(tf.float32, [None, 1], "rewards")
        self.next_state_ph = tf.placeholder(
            tf.float32, [None, state_dim], "next_states")
        self.termination_ph = tf.placeholder(
            tf.float32, [None, 1], 'terminations')
        self.fictional_memory = GeneralMemory(
            memory_size, state_dim, 0, 1, state_dim, 1)

    def initializeImagination(self, layer_dims=[30, 30]):
        with tf.variable_scope("imagination"):
            flow = self.state
            for i, size in enumerate(layer_dims):
                flow = fullyConnected("layer%i" % i, flow, size, tf.nn.relu, .003)

            predict_state = fullyConnected(
                "state_prediction", flow, self.action_dim * self.state_dim, initializer=.003)
            predict_state = tf.reshape(
                predict_state, (-1, self.action_dim, self.state_dim))
            predict_reward = fullyConnected(
                "reward_prediction", flow, self.action_dim, initializer=.003)
            predict_reward = tf.reshape(
                predict_reward, (-1, self.action_dim, 1))
            predict_termination = fullyConnected(
                "termination_prediction", flow, self.action_dim, activation=tf.nn.sigmoid, initializer=.003)
            predict_termination = tf.reshape(
                predict_termination, (-1, self.action_dim, 1))

        indexes = tf.stack(
            [tf.range(0, tf.shape(flow)[0]), self.action_ph], axis=1)
        self.predict_state = tf.gather_nd(predict_state, indexes)
        self.predict_reward = tf.gather_nd(predict_reward, indexes)
        self.predict_termination = tf.gather_nd(predict_termination, indexes)

        self.imagination_loss = tf.reduce_mean(tf.square(
            self.next_state_ph - self.predict_state)) +\
            tf.reduce_mean(tf.square(self.predict_reward - self.reward_ph))

        self.imagination_train_op = tf.train.AdamOptimizer(.01).minimize(
            self.imagination_loss)

    def imagine(self, session, batch, step=1):
        state = self.memory.sample(batch)[0]
        for _ in range(step):
            action = np.argmax(session.run(
                self.action_value, {self.state: state}), axis=1)

            reward, next_state, termination = session.run([self.predict_reward, self.predict_state, self.predict_termination], {
                self.state: state, self.action_ph: action})
            self.fictional_memory.addBatch(
                state, action, reward, next_state, termination)
            state = next_state

    def trainImagination(self, session, batch):
        state, action, reward, next_state, termination = self.memory.sample(
            batch)
        imagination_loss, _ = session.run([self.imagination_loss, self.imagination_train_op], {
            self.state: state,
            self.action_ph: action,
            self.next_state_ph: next_state,
            self.reward_ph: reward,
            self.termination_ph: termination[:, None]})
        return imagination_loss

    def train(self, session, batch, discount=.97, step=1):
        state, action, reward, next_state, termination = self.memory.sample(
            batch)
        imagination_loss, _ = session.run([self.imagination_loss, self.imagination_train_op], {
            self.state: state,
            self.action_ph: action,
            self.next_state_ph: next_state,
            self.reward_ph: reward,
            self.termination_ph: termination[:, None]})

        next_state_value = session.run(
            self.target_action_value, {self.state: next_state})
        observed_value = reward + discount * \
            np.max(next_state_value, 1, keepdims=True)
        observed_value[termination] = reward[termination]# / (1 - discount)

        _, loss = session.run([self.train_op, self._loss], {
            self.state: state, self.action_ph: action, self.action_value_ph:
            observed_value[:, 0]})

        for _ in range(step):
            state, action, reward, next_state, termination = self.fictional_memory.sample(
                batch)
            next_state_value = session.run(
                self.target_action_value, {self.state: next_state})
            observed_value = reward + discount * \
                np.max(next_state_value, 1, keepdims=True)
            observed_value = observed_value * \
                (1 - termination) + reward * termination# / (1 - discount)

            _, l = session.run([self.train_op, self._loss], {
                self.state: state, self.action_ph: action, self.action_value_ph: observed_value[:, 0]})
            loss += l

        session.run(self.update_op)
        return loss / (step + 1), imagination_loss


class ImaginationDQN2(ImaginationDQN):
    def initializeImagination(self, before_action=[30], after_action=[30]):
        indexes = tf.stack(
            [tf.range(0, tf.shape(self.action_ph)[0]), self.action_ph], axis=1)

        with tf.variable_scope("imagination"):
            net = self.state
            for i, size in enumerate(before_action[:-1]):
                net = fullyConnected("layer_0", net, size,
                                     tf.nn.relu, initializer=.003)
            net = fullyConnected("layer_0", net, self.action_dim *
                                 before_action[-1], tf.nn.relu, initializer=.003)
            net = tf.reshape(net, (-1, self.action_dim, before_action[-1]))
            net = tf.gather_nd(net, indexes)

            for i, size in enumerate(after_action):
                net = fullyConnected("layer%i" %
                                     (i + 1), net, size, tf.nn.relu, .003)

            self.predict_state = fullyConnected(
                "state_prediction", net, self.state_dim, initializer=.003)
            self.predict_reward = fullyConnected(
                "reward_prediction", net, 1, initializer=.003)
            self.predict_termination = fullyConnected(
                "termination_prediction", net, 1, activation=tf.nn.sigmoid, initializer=.003)

        self.imagination_loss = tf.reduce_mean(tf.square(
            self.next_state_ph - self.predict_state)) +\
            tf.reduce_mean(tf.square(self.predict_reward - self.reward_ph)) +\
            tf.reduce_mean(
                tf.square(self.predict_termination - self.termination_ph))

        self.imagination_train_op = tf.train.AdamOptimizer(.01).minimize(
            self.imagination_loss)



class HighDimImaginationDQN(ImaginationDQN):
    def __init__(self, reading_dim, pose_dim, action_dim, memory_size):
        self.action_dim = action_dim
        self.reading_dim = reading_dim
        self.memory_size = memory_size

        self.reading_ph = tf.placeholder(
            tf.float32, [None, reading_dim], "reading")
        self.pose_ph = tf.placeholder(tf.float32, [None, pose_dim], "pose")
        self.action_ph = tf.placeholder(tf.int32, [None], "actions")
        self.reward_ph = tf.placeholder(tf.float32, [None, 1], "rewards")
        self.next_reading_ph = tf.placeholder(
            tf.float32, [None, reading_dim], "next_reading")
        self.next_pose_ph = tf.placeholder(
            tf.float32, [None, pose_dim], "next_pose")
        self.termination_ph = tf.placeholder(
            tf.float32, [None, 1], 'terminations')

        self.action_value_ph = tf.placeholder(
            tf.float32, [None], "action_values")

        self.memory = GeneralMemory(
            memory_size, reading_dim, pose_dim, 0, 1, reading_dim, pose_dim, -1)

    def initializeImagination(self, layer_dims = [30,30]):
        # initializer = tf.random_uniform_initializer(-0.005, 0.005)
        # stride = (1, 3)
        # patch = (1, 5)
        # depth = (4, 4, 8)

        # depth = (1,) + depth

        # shapes = [[1, self.reading_dim]]
        # for i in range(len(depth)):
        #     next_shape = [-(-shapes[i][0] // stride[0]), -
        #                   (-shapes[i][1] // stride[1])]
        #     shapes.append(next_shape)

        # print("Conv dimensions are", shapes)
        # shapes = list(reversed(shapes))

        # conv_params = []
        # for i in range(len(depth[:-1])):
        #     w = tf.get_variable("conv_weight_%i" % i, shape=patch + (
        #         depth[i], depth[i + 1]), initializer=initializer, dtype=tf.float32)
        #     b = tf.Variable(tf.constant(.01, shape=[depth[i + 1]]))
        #     conv_params.append((w, b))

        # deconv_params = []
        # for i in range(len(depth[:-1])):
        #     w = tf.get_variable("deconv_weight_%i" % i, shape=patch + (
        #         depth[i], depth[i + 1]), initializer=initializer, dtype=tf.float32)
        #     b = tf.Variable(tf.constant(.01, shape=[depth[i]]))
        #     deconv_params.append((w, b))
        # deconv_params = list(reversed(deconv_params))

        # def _makeConv(input):
        #     flow = input
        #     for w, b in conv_params:
        #         print("in conv", flow.get_shape(), w.get_shape())
        #         flow = tf.nn.conv2d(flow, w, [1, 1, 1, 1], padding='SAME') + b
        #         flow = tf.nn.max_pool(
        #             flow, (1,) + stride + (1,), (1,) + stride + (1,), padding='SAME')
        #         flow = tf.nn.relu(flow)
        #     return flow

        # def _makeDeconv(input):
        #     flow = input
        #     shape = input.get_shape().as_list()
        #     batch_size, height, width, depth = shape
        #     batch_size = tf.shape(input)[0]
        #     for i, p in enumerate(deconv_params):
        #         w, b = p
        #         next_depth, depth = w.get_shape().as_list()[-2:]
        #         height, width = shapes[i + 2]
        #         print("hw", height, width)
        #         flow = tf.nn.conv2d_transpose(flow, w, strides=(1,) + stride + (1,), output_shape=[
        #                                       batch_size, height, width, next_depth], padding="SAME")
        #         print("deconv shape", [batch_size, height, width, next_depth])
        #     return flow

        # batch_size = tf.shape(self.reading_ph)[0]

        # encoded_1 = _makeConv(tf.reshape(
        #     self.reading_ph, [batch_size, 1, 150, 1]))
        # deconved_1 = tf.squeeze(_makeDeconv(encoded_1))
        # loss_1 = tf.reduce_mean(tf.square(self.reading_ph - deconved_1))

        # encoded_2 = _makeConv(tf.reshape(
        #     self.next_reading_ph, [batch_size, 1, 150, 1]))
        # deconved_2 = tf.squeeze(_makeDeconv(encoded_2))
        # loss_2 = tf.reduce_mean(
        #     tf.square(self.next_reading_ph - deconved_2))

        # self.deconved = deconved_1
        # self.next_deconved = deconved_2

        # shape = encoded_1.get_shape().as_list()
        # encoded_1_flat = tf.reshape(encoded_1, (-1, shape[3] * shape[2]))
        # shape = encoded_2.get_shape().as_list()
        # encoded_2_flat = tf.reshape(encoded_2, (-1, shape[3] * shape[2]))
        # self.encoded = encoded_1_flat
        # self.next_encoded = encoded_2_flat
        # print(shape)

        # self.encoded_ph = tf.placeholder(
        #     tf.float32, [shape[0], shape[2] * shape[3]], name='encoded')
        # self.next_encoded_ph = tf.placeholder(
        #     tf.float32, [shape[0], shape[2] * shape[3]], name="next_encoded")
        # self.state = tf.concat([self.encoded_ph, self.pose_ph], 1)
        # self.next_state = tf.concat(
        #     [self.next_encoded_ph, self.next_pose_ph], 1)
        self.state = self.pose_ph
        self.next_state = self.next_pose_ph
        
        state_dim = self.state.get_shape().as_list()[1]
        self.fictional_memory = GeneralMemory(
            self.memory_size, state_dim, 0, 1, state_dim, 1)

        print("state shape", state_dim)


        with tf.variable_scope("imagination"):
            flow = self.state
            for i, size in enumerate(layer_dims):
                flow = fullyConnected("layer%i" % i, flow, size, tf.nn.relu, .003)

            predict_state = fullyConnected(
                "state_prediction", flow, self.action_dim * state_dim, initializer=.003)
            predict_state = tf.reshape(
                predict_state, (-1, self.action_dim, state_dim))
            predict_reward = fullyConnected(
                "reward_prediction", flow, self.action_dim, initializer=.003)
            predict_reward = tf.reshape(
                predict_reward, (-1, self.action_dim, 1))
            predict_termination = fullyConnected(
                "termination_prediction", flow, self.action_dim, activation=tf.nn.sigmoid, initializer=.003)
            predict_termination = tf.reshape(
                predict_termination, (-1, self.action_dim, 1))

        indexes = tf.stack(
            [tf.range(0, tf.shape(flow)[0]), self.action_ph], axis=1)
        self.predict_state = tf.gather_nd(predict_state, indexes)
        self.predict_reward = tf.gather_nd(predict_reward, indexes)
        self.predict_termination = tf.gather_nd(predict_termination, indexes)

        # self.conv_loss = loss_1 + loss_2
        # self.conv_train_op = tf.train.AdamOptimizer(
        #     .01).minimize(self.conv_loss)

        self.imagination_single_loss = tf.square(
            self.next_state - self.predict_state)
        self.imagination_loss = tf.reduce_mean(self.imagination_single_loss) +\
            tf.reduce_mean(tf.square(self.predict_reward - self.reward_ph)) +\
            tf.reduce_mean(
                tf.square(self.predict_termination - self.termination_ph))

        self.imagination_train_op = tf.train.AdamOptimizer(.01).minimize(
            self.imagination_loss)

    def imagine(self, session, batch, step=1):
        if step == 0:
            return
        reading, pose = self.memory.sample(batch)[:2]
        # enc = session.run(self.encoded, {self.reading_ph: reading})

        state = session.run(
            self.state, {self.pose_ph: pose})
        for _ in range(step):
            action = np.argmax(session.run(
                self.action_value, {self.state: state}), axis=1)

            reward, next_state, termination = session.run([self.predict_reward,
                                                           self.predict_state,
                                                           self.predict_termination], {
                self.state: state,
                self.action_ph: action})
            self.fictional_memory.addBatch(
                state, action, reward, next_state, termination)
            state = next_state
        pass

    def train(self, session, batch=None, discount=.97, step=0):
        reading, pose, action, reward, next_reading, next_pose, termination = self.memory.sample(
            batch)

        # _, conv_loss, enc_1, enc_2 = session.run([self.conv_train_op, self.conv_loss, self.encoded, self.next_encoded], {
        #     self.reading_ph: reading, self.next_reading_ph: next_reading})

        imagination_loss, _ = session.run([self.imagination_loss, self.imagination_train_op], {
            self.pose_ph: pose,
            self.action_ph: action,
            self.next_pose_ph: next_pose,
            self.reward_ph: reward,
            self.termination_ph: termination[:, None]})

        next_state_value = session.run(
            self.target_action_value, {self.pose_ph: next_pose})
        observed_value = reward + discount * \
            np.max(next_state_value, 1, keepdims=True)
        observed_value[termination] = reward[termination] / (1 - discount)

        _, loss = session.run([self.train_op, self._loss], {
            self.pose_ph: pose,
            self.action_ph: action,
            self.action_value_ph: observed_value[:, 0]})


        # next_state_value = session.run(
        #     self.target_action_value, {self.encoded_ph: enc_2, self.pose_ph: next_pose})
        # observed_value = reward + discount * \
        #     np.max(next_state_value, 1, keepdims=True)
        # observed_value[termination] = reward[termination]  # / (1 - discount)

        # _, loss = session.run([self.train_op, self._loss], {
        #     self.encoded_ph: enc_1,
        #     self.pose_ph: pose,
        #     self.action_ph: action,
        #     self.action_value_ph: observed_value[:, 0]})

        for _ in range(step):
            state, action, reward, next_state, termination = self.fictional_memory.sample(
                batch)
            next_state_value = session.run(
                self.target_action_value, {self.state: next_state})
            observed_value = reward + discount * \
                np.max(next_state_value, 1, keepdims=True)
            observed_value = observed_value * \
                (1 - termination) + reward * termination / (1 - discount)

            _, l = session.run([self.train_op, self._loss], {
                self.state: state, self.action_ph: action, self.action_value_ph: observed_value[:, 0]})
            loss += l

        return loss / (step + 1), imagination_loss,0# np.mean(conv_loss)

    def policy(self, session, reading, pose):
        # encoded = session.run(self.encoded, {self.reading_ph: [reading]})
        return session.run(self.action_value, {self.pose_ph: [pose]})[0]

    def memorize(self, reading, pose, action, reward, next_reading, next_pose, terminal):
        self.memory.add(reading, pose, action, reward,
                        next_reading, next_pose, terminal)


class VisualDQN(ImaginationDQN):
    def __init__(self, frame_shape, action_dim, memory_size):
        self.action_dim = action_dim
        state_dim = 51
        frame_shape = list(frame_shape)
        self.frame_shape = frame_shape
        frame_shape[2] = 1
        self.frame_ph = tf.placeholder(
            tf.float32, [None] + frame_shape[:2], "frame")
        self.action_ph = tf.placeholder(tf.int32, [None], "actions")
        self.reward_ph = tf.placeholder(tf.float32, [None, 1], "rewards")
        self.next_frame_ph = tf.placeholder(
            tf.float32, [None] + frame_shape[:2], "next_frame")
        self.termination_ph = tf.placeholder(
            tf.float32, [None, 1], 'terminations')

        self.action_value_ph = tf.placeholder(
            tf.float32, [None], "action_values")

        self.memory = GeneralMemory(
            memory_size, frame_shape[:-1], 0, 1, frame_shape[:-1], -1)
        self.fictional_memory = GeneralMemory(
            memory_size, state_dim, 0, 1, state_dim, 1)

    def initializeImagination(self, before_action=[30], after_action=[30]):
        initializer = tf.random_uniform_initializer(-0.005, 0.005)
        stride = (2, 2)
        patch = (4, 4)
        depth = (64, 128, 128, 128)

        depth = (1,) + depth

        shapes = [self.frame_shape[:2]]
        for i in range(len(depth)):
            next_shape = [-(-shapes[i][0] // stride[0]), -
                          (-shapes[i][1] // stride[1])]
            shapes.append(next_shape)

        print("Conv dimensions are", shapes)
        shapes = list(reversed(shapes))

        conv_params = []
        for i in range(len(depth[:-1])):
            w = tf.get_variable("conv_weight_%i" % i, shape=patch + (
                depth[i], depth[i + 1]), initializer=initializer, dtype=tf.float32)
            b = tf.Variable(tf.constant(.01, shape=[depth[i + 1]]))
            conv_params.append((w, b))

        deconv_params = []
        for i in range(len(depth[:-1])):
            w = tf.get_variable("deconv_weight_%i" % i, shape=patch + (
                depth[i], depth[i + 1]), initializer=initializer, dtype=tf.float32)
            b = tf.Variable(tf.constant(.01, shape=[depth[i]]))
            deconv_params.append((w, b))
        deconv_params = list(reversed(deconv_params))

        def _makeConv(input):
            flow = input
            for w, b in conv_params:
                print("in conv", flow.get_shape(), w.get_shape())
                flow = tf.nn.conv2d(flow, w, [1, 1, 1, 1], padding='SAME') + b
                flow = tf.nn.max_pool(
                    flow, (1,) + stride + (1,), (1,) + stride + (1,), padding='SAME')
                flow = tf.nn.relu(flow)
            return flow

        def _makeDeconv(input):
            flow = input
            shape = input.get_shape().as_list()
            batch_size, height, width, depth = shape
            batch_size = tf.shape(input)[0]
            for i, p in enumerate(deconv_params):
                w, b = p
                next_depth, depth = w.get_shape().as_list()[-2:]
                height, width = shapes[i + 2]
                print("hw", height, width)
                flow = tf.nn.conv2d_transpose(flow, w, strides=(1,) + stride + (1,), output_shape=[
                                              batch_size, height, width, next_depth], padding="SAME")
                print("deconv shape", [batch_size, height, width, next_depth])
            return flow

        batch_size = tf.shape(self.frame_ph)[0]
        frame = tf.expand_dims(self.frame_ph, 3)
        next_frame = tf.expand_dims(self.next_frame_ph, 3)
        encoded_1 = _makeConv(frame)
        deconved_1 = _makeDeconv(encoded_1)
        loss_1 = tf.reduce_mean(tf.square(frame - deconved_1))
        self.deconved_1 = deconved_1

        encoded_2 = _makeConv(next_frame)
        deconved_2 = _makeDeconv(encoded_2)
        loss_2 = tf.reduce_mean(
            tf.square(next_frame - deconved_2))
        self.deconved_2 = deconved_2

        self.conv_loss = loss_1 + loss_2
        self.conv_train_op = tf.train.AdamOptimizer(
            .001).minimize(self.conv_loss)

        shape = encoded_1.get_shape().as_list()
        print("encoded dims", shape)
        encoded_1_flat = tf.reshape(
            encoded_1, (-1, shape[3] * shape[2] * shape[1]))
        self.state = encoded_1_flat

        encoded_2_flat = tf.reshape(
            encoded_2, (-1, shape[3] * shape[2] * shape[1]))
        self.next_state = encoded_2_flat

        self.state_dim = self.state.get_shape().as_list()[1]
        print("state dim", self.state_dim)

        self.state_ph = tf.placeholder(
            tf.float32, [shape[0], shape[1] * shape[2] * shape[3]], name='encoded')
        self.next_state_ph = tf.placeholder(
            tf.float32, [shape[0], shape[1] * shape[2] * shape[3]], name="next_encoded")

        indexes = tf.stack(
            [tf.range(0, tf.shape(self.action_ph)[0]), self.action_ph], axis=1)

        with tf.variable_scope("imagination"):
            net = self.state_ph
            for i, size in enumerate(before_action[:-1]):
                net = fullyConnected("before_action_layer_%i" % i, net, size,
                                     tf.nn.relu, initializer=.003)
            net = fullyConnected("layer_0", net, self.action_dim *
                                 before_action[-1], tf.nn.relu, initializer=.003)
            net = tf.reshape(net, (-1, self.action_dim, before_action[-1]))
            net = tf.gather_nd(net, indexes)

            for i, size in enumerate(after_action):
                net = fullyConnected("layer%i" %
                                     (i + 1), net, size, tf.nn.relu, .003)

            self.predict_state = fullyConnected(
                "state_prediction", net, self.state_dim, initializer=.003)
            self.predict_reward = fullyConnected(
                "reward_prediction", net, 1, initializer=.003)
            self.predict_termination = fullyConnected(
                "termination_prediction", net, 1, activation=tf.nn.sigmoid, initializer=.003)

        self.imagination_single_loss = tf.square(
            self.next_state_ph - self.predict_state)
        self.imagination_loss = tf.reduce_mean(self.imagination_single_loss) +\
            tf.reduce_mean(tf.square(self.predict_reward - self.reward_ph)) +\
            tf.reduce_mean(
                tf.square(self.predict_termination - self.termination_ph))

        self.imagination_train_op = tf.train.AdamOptimizer(.001).minimize(
            self.imagination_loss)

    def imagine(self, session, batch, step=1):
        if step < 10:
            return
        reading, pose = self.memory.sample(batch)[:2]
        enc = session.run(self.state, {self.reading_ph: reading})

        state = session.run(
            self.state, {self.state_ph: enc, self.pose_ph: pose})
        for _ in range(step):
            action = np.argmax(session.run(
                self.action_value, {self.state: state}), axis=1)

            reward, next_state, termination = session.run([self.predict_reward,
                                                           self.predict_state,
                                                           self.predict_termination], {
                self.state: state,
                self.action_ph: action})
            self.fictional_memory.addBatch(
                state, action, reward, next_state, termination)
            state = next_state
        pass

    def train(self, session, batch=None, discount=.97, step=0):
        state, action, reward, next_state, termination = self.memory.sample(
            batch)
        # reading = reading/20
        # next_reading = next_reading/20

        if step:
            r = session.run([self.deconved_1, self.deconved_2], {
                            self.frame_ph: state, self.next_frame_ph: next_state})
            print(np.mean(state - r[0][:, :, :, 0]),
                  np.mean(next_state - r[1][:, :, :, 0]))
            print(np.mean(next_state - r[1][:, :, :, 0]),
                  np.mean(state - r[0][:, :, :, 0]))
            fig = figure()
            a = fig.add_subplot(1, 3, 1)
            imshow(state[0], cmap='Greys_r')

            a = fig.add_subplot(1, 3, 2)
            imshow(r[0][0, :, :, 0], cmap='Greys_r')

            a = fig.add_subplot(1, 3, 3)
            imshow(state[0] - r[0][0, :, :, 0], cmap='Greys_r')
            show()
            raise

        _, conv_loss, enc_1, enc_2 = session.run([self.conv_train_op, self.conv_loss, self.state, self.next_state], {
                                                 self.frame_ph: state, self.next_frame_ph: next_state})
        # print(enc_1)
        imagination_loss, _ = session.run([self.imagination_loss, self.imagination_train_op], {
            self.state_ph: enc_1,
            self.action_ph: action,
            self.next_state_ph: enc_2,
            self.reward_ph: reward,
            self.termination_ph: termination[:, None]})

        next_state_value = session.run(
            self.target_action_value, {self.state_ph: enc_2})
        observed_value = reward + discount * \
            np.max(next_state_value, 1, keepdims=True)
        observed_value[termination] = reward[termination]  # / (1 - discount)

        _, loss, sl = session.run([self.train_op, self._loss, self.single_loss], {
            self.state_ph: enc_1,
            self.action_ph: action,
            self.action_value_ph: observed_value[:, 0]})

        for _ in range(step):
            state, action, reward, next_state, termination = self.fictional_memory.sample(
                batch)
            next_state_value = session.run(
                self.target_action_value, {self.state: next_state})
            observed_value = reward + discount * \
                np.max(next_state_value, 1, keepdims=True)
            observed_value = observed_value * \
                (1 - termination) + reward * termination  # / (1 - discount)

            _, l = session.run([self.train_op, self._loss], {
                self.state: state, self.action_ph: action, self.action_value_ph: observed_value[:, 0]})
            loss += l

        # session.run(self.update_op)
        return loss / (step + 1), imagination_loss, np.mean(conv_loss)

    def initialize(self, layer_dims=[30, 30], optimizer=None):
        def _make():
            flow = self.state_ph
            for i, size in enumerate(layer_dims):
                flow = fullyConnected(
                    "layer%i" % i, flow, size, tf.nn.relu, initializer=.003)

            return fullyConnected(
                "output_layer", flow, self.action_dim, initializer=.003)

        with tf.variable_scope('learner'):
            self.action_value = _make()
        with tf.variable_scope('target'):
            self.target_action_value = _make()

        self.update_op = copyScopeVars('learner', 'target')

        row = tf.range(0, tf.shape(self.action_value)[0])
        indexes = tf.stack([row, self.action_ph], axis=1)
        action_value = tf.gather_nd(self.action_value, indexes)

        self.single_loss = tf.square(action_value - self.action_value_ph)
        self._loss = tf.reduce_mean(self.single_loss)

        self.optimizer = optimizer or tf.train.AdamOptimizer(.001)

        # parameters = getScopeParameters('learner')
        # grads = tf.gradients(self._loss, parameters)
        # self.train_op = self.optimizer.apply_gradients(zip(grads, parameters))

        self.train_op = self.optimizer.minimize(self._loss)

    def policy(self, session, frame):
        encoded = session.run(self.state, {self.frame_ph: [frame]})
        return session.run(self.action_value, {self.state_ph: encoded})[0]

    def memorize(self, frame, action, reward, next_frame, terminal):
        self.memory.add(frame, action, reward, next_frame, terminal)
