import os
import shutil
import inspect


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


if __name__ == "__main__":
    here()
