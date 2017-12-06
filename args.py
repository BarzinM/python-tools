import argparse as ap


def getFlags():
    parser = ap.ArgumentParser()
    parser.add_argument('positional', choices=['spam', 'spam2'])
    parser.add_argument('--optional', choices=['foo1', 'foo2'])
    args = parser.parse_args()

    return args


class Args(object):
    def __init__(self):
        self.parser = ap.ArgumentParser()
        pass

    def positional(self, *args, **kwargs):
        pass

    def options(self, *args, **kwargs):
        for arg in args:
            self.parser.add_argument(arg,action='store_true')

        for kwa in kwargs:
            self.parser.add_argument(kwa, default=kwargs[kwa])

    def parse(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    a = Args()
    a.options('double', 'hind', this='that')
    args = a.parse()
    if args.double:
        print('DOUBLE')
