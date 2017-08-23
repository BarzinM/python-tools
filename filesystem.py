import os
import shutil
import inspect
import glob


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
