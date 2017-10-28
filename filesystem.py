import os
import shutil
import inspect
import glob
import __main__


def mainFileName(full=False):
    path = __main__.__file__
    path = os.path.split(path)[1]
    if not full:
        path = os.path.splitext(path)[0]
    return path


def mainFilePath():
    import __main__
    return os.path.split(__main__.__file__)[0]


def mkd(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


if __name__ == "__main__":
    print(here())
    print(ls(here()))
