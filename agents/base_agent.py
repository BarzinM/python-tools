import tensorflow as tf


class BaseModel(object):

    def __init__(self):
        self.train_count = 0
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.train.get_or_create_global_step()
            self._build()
            init = tf.global_variables_initializer()
        self.session = tf.Session(graph=self.graph)
        self.session.run(init)
        self.saver = None

    def save(self, path, step=None):
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=5)

        if step is None:
            if self.train_count == 0:
                raise ValueError(
                    "Either step argument should be provided, or "
                    "`BaseModel.train_count` should be incremented.")
            else:
                step = self.train_count

        path = self.saver.save(self.session, path, global_step=step)
        print("Saved model to", path)

    def restor(self, path):
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=5)

        self.saver.restore(self.session, path)

    def close(self):
        self.session.close()


class BaseAgent(BaseModel):

    def __init__(self, state_dim, action_dim):
        pass
