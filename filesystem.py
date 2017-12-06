import os
import sys
import shutil
import inspect
import glob
import __main__


class FileReport(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.handle = open(file_name, 'w', 1)

    def add(self, *args):
        text = '{} ' * len(args) + '\n'
        self.handle.write(text.format(*args))

    def close(self):
        close(self.handle)


def tailNumber(text):
    st = ""
    for c in text[::-1]:
        if c.isdigit():
            st = c + st
        else:
            break
    return st, text[:-len(st)]


def incrementName(path):
    number, raw_name = tailNumber(path)
    if len(number):
        new_name = raw_name + str(int(number) + 1)
    else:
        new_name = path + '_0'
    return new_name


def mainFileName(full=False):
    path = __main__.__file__
    path = os.path.split(path)[1]
    if not full:
        path = os.path.splitext(path)[0]
    return path


def mainFilePath():
    import __main__
    return os.path.split(__main__.__file__)[0]


def mkd(path, increment=True):
    if path[-1] == '/':
        path = path[:-1]

    if increment:
        temp_path = path
        i = 0
        while os.path.exists(temp_path) and len(os.listdir(temp_path)) > 0:
            # path = incrementName(path + path_2)
            temp_path = path + "_%i" % i
            i += 1
        path = temp_path
        mkd(path, increment=False)
    elif not os.path.exists(path):
        os.makedirs(path)
    return path


def rmd(path):
    shutil.rmtree(path)


def rm(path):
    os.remove(path)


def here():
    frame = inspect.stack()
    module = inspect.getmodule(frame[1][0])
    return os.path.dirname(os.path.realpath(module.__file__))


def ls(dir, extention='*.*'):
    return glob.glob(os.path.join(dir, extention))


class Arguments(object):
    def __init__(self):
        self.flags = {}
        self.options = {}

    def flag(self, *args, **kwargs):
        self.flag.update(kwargs)
        for arg in args:
            self.flags[arg] = False

    def options(self, *args, **kwargs):
        self.options.update(kwargs)
        for arg in args:
            self.options[arg] = 0

    def parse(self):
        self.argv = sys.argv[1:]

    def __call__(self, arg):
        if arg in self.options:
            return self.options[arg]
        elif arg in self.flags:
            return self.flags[arg]


if __name__ == "__main__":
    # print(here())
    # print(ls(here()))
    a = Arguments()
    a.flag('verbose')
    print(a.flags)
    a.parse()
