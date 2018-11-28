import numpy as np
import glob


class DataLog(object):

    def __init__(self, path, shape=None, dtype=np.float, resume=False):
        self.path = path
        self.file_counter = 0
        if shape is not None:
            self.pointer = 0
            self.array = np.zeros(shape, dtype=dtype)
            self.max_len = shape[0]
            if resume:
                files = self._available_files()
                self.file_counter = max(files) + 1

    def __iter__(self):
        files = glob.glob(self.path + "_*.npy")
        for f in files:
            yield np.load(f)

    def _available_files(self):
        files = glob.glob(self.path + "_*.npy")
        files = [int(f[len(self.path) + 1:-4]) for f in files]
        return files

    def next_file_name(self):
        path = self.path + "_%i" % self.file_counter
        self.file_counter += 1

        return path

    def add(self, *values):
        self.array[self.pointer] = values
        self.pointer += 1
        if self.pointer >= self.max_len:
            self.save()

    def save(self):
        if self.pointer:
            path = self.next_file_name()
            np.save(path, self.array[:self.pointer])
            self.pointer = 0


if __name__ == "__main__":
    d = DataLog("temp", (5, 2), resume=True)
    d.add(1, 2)
    d.add(1, 2)
    d.add(1, 2)
    d.add(1, 2)
    d.add(1, 2)
    d.add(1, 2)
    d.add(1, 2)

    for a in d:
        print(a)
