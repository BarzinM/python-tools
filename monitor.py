import tensorflow as tf
from sys import stdout
import os
import shutil
import __main__


class Monitor(object):
    def __init__(self, path=None, session=None):
        if path is None:
            path = 'results/' + __main__.__file__[:-3]
        if os.path.isdir(path):
            shutil.rmtree(path)
        if session is None:
            self.writer = tf.summary.FileWriter(path)
        else:
            self.writer = tf.summary.FileWriter(path, self.session.graph)
        self.variables = {}
        self.iteration = 0

    def addVar(self, name, initial=0.):
        self.var = tf.Variable(initial)
        tf.summary.scalar(name, self.var)
        self.variables[self.var] = initial
        return self.var

    def setup(self):
        self.op = tf.summary.merge_all()

    def set(self, var, value):
        self.variables[var] = value

    def update(self, session, i=None):
        summary = session.run(self.op, self.variables)
        if i is None:
            self.iteration += 1
            self.writer.add_summary(summary, self.iteration)
        else:
            self.writer.add_summary(summary, i)
        self.writer.flush()


class Display(object):
    def __init__(self):
        self.lines = 0

    def print(self, *args):
        print(*args)
        self.lines += 1

    def clear(self, title=None):
        stdout.write("\033[F\033[K" * self.lines)
        self.new(title)

    def new(self, seperator=None):
        if seperator:
            print(seperator)
        self.lines = 0


if __name__ == "__main__":
    from time import sleep
    disp = Display()
    for i in range(10):
        disp.clear()
        sleep(1.)
        disp.print("this", i)
        disp.print("%i" % (10 - 2 * i))
    disp.new("----------")
    for i in range(5):
        disp.print("this %i" % i)
        disp.clear()
        sleep(.1)
