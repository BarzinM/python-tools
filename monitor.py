import tensorflow as tf
from sys import stdout
import os
import __main__
from filesystem import mainFileName


class Monitor(object):
    def __init__(self, path=None, sub='', session=None, verbose=True):
        if path is None:
            path = 'summary/' + mainFileName()
        path = path + '/' + sub
        if os.path.isdir(path):
            import shutil
            shutil.rmtree(path)
        if verbose:
            print("Saving monitor files to", path)
        if session is None:
            self.writer = tf.summary.FileWriter(path)
        else:
            self.writer = tf.summary.FileWriter(path, session.graph)
        self.variables = {}
        self.strings = []
        self.iteration = 0
        self.summaries = []
        self.global_step = tf.train.get_or_create_global_step()

    def scalar(self, iterable):
        assert type(iterable) in [list, tuple, dict]
        if type(iterable) in [list, tuple]:
            iterable = {item.name: item for item in iterable}
        for name, var in iterable.items():
            if type(var) != tf.Variable:
                var = tf.Variable(var, name=name)
            self.summaries.append(tf.summary.scalar(name, var))
        self.op = tf.summary.merge(self.summaries)

    def histogram(self, iterable):
        if type(iterable) in [list, tuple]:
            iterable = {item.name: item for item in iterable}
        for name, var in iterable.items():
            if type(var) != tf.Variable:
                var = tf.Variable(var, name=name)
            self.summaries.append(tf.summary.histogram(name, var))
        self.op = tf.summary.merge(self.summaries)

    def update(self, session, dictionary={}, i=None):
        availables = self.variables.keys()
        new_d = {}
        for name, value in dictionary.items():
            if name not in availables:
                var = tf.Variable(value, name=name)
                self.variables[name] = var
                self.summaries.append(
                    tf.summary.scalar(name, self.variables[name]))
                self.op = tf.summary.merge(self.summaries)
            new_d[self.variables[name]] = value

        # new_d = {self.variables[d]: dictionary[d] for d in dictionary.keys()}

        if i is None:
            summary, step = session.run(
                [self.op, self.global_step], feed_dict=new_d)
            # self.iteration += 1
            self.writer.add_summary(summary, step)
        else:
            summary = session.run(self.op, feed_dict=new_d)
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


def breakLine(text, wrap=80):
    if len(text) > wrap:
        char = wrap
        while char > 0 and text[char] != ' ':
            char -= 1
        if char:
            text = [text[:char]] + breakLine(text[char + 1:], wrap)
        else:
            text = [text[:wrap - 1] + '-'] + breakLine(text[wrap - 1:], wrap)
        return text
    else:
        return [cleanLine(text)]


def cleanLine(text):
    if text[-1] == ' ':
        text = text[:-1]
    elif text[0] == ' ':
        text = text[1:]
    else:
        return text
    return cleanLine(text)


def boxPrint(text, wrap=0):
    line_style = '-'
    paragraph = text.split('\n')
    if wrap > 0:
        index = 0
        while index < len(paragraph):
            paragraph[index] = cleanLine(paragraph[index])
            if len(paragraph[index]) > wrap:
                paragraph = paragraph[:index] + \
                    breakLine(paragraph[index], wrap) + paragraph[index + 1:]
            index += 1

    length = (max([len(line) for line in paragraph]))
    print('+' + line_style * length + '+')
    for line in paragraph:
        print('|' + line + ' ' * (length - len(line)) + '|')
    print('+' + line_style * length + '+')


if __name__ == "__main__":
    text = "Some text comes here to be printed in a box!!!"
    boxPrint(text, 20)
    text = "Title:\nBody lorem ipsum something body\ncheers,"
    boxPrint(text, 20)
    boxPrint(text)
    text = "No Space:\nTextHasNoSpaceForWrappingWhichGetsBrokenUsingDashes"
    boxPrint(text, 20)


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
