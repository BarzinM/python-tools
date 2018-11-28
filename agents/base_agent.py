import tensorflow as tf


class BaseModel(object):

    def build(self):
        self.train_count = 0
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.train.get_or_create_global_step()
            self._build()
            init = tf.global_variables_initializer()
        self.session = tf.Session(graph=self.graph)
        self.session.run(init)
        self._saver = None

    def saver(self, max_to_keep=5):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=max_to_keep)

    def save(self, path, step=None):
        self.saver()
        if step is None:
            if self.train_count == 0:
                raise ValueError(
                    "Either step argument should be provided, or "
                    "`BaseModel.train_count` should be incremented.")
            else:
                step = self.train_count

        path = self._saver.save(self.session, path, global_step=step)
        print("Saved model to", path)

    def restor(self, path):
        self.saver()

        self._saver.restore(self.session, path)

    def close(self):
        self.session.close()


# class BaseAgent(BaseModel):

#     def __init__(self):
#         pass
