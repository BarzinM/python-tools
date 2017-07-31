from numpy import empty, zeros, copy, sum, power
from numpy.random import randint, seed, choice


class Buffer(object):
    def __init__(self, length, shape):
        self.buffer = empty((length, *shape))
        self.pointer = 0
        self.legnth = length
        self.add = self._fill
        self.sample = self._sampleFromHalfFull
        self.filled = False
        self._count = 0

    def _fill(self, a):
        self.buffer[self.pointer] = a
        self.pointer += 1
        self._count += 1
        if self.pointer == self.legnth:
            self._switch()

    def count(self):
        return self._count

    def _switch(self):
        self.pointer = 0
        self.filled = True
        self.add = self._update
        self.sample = self._sampleFromFull

    def _update(self, a):
        self.buffer[self.pointer] = a
        self.pointer = (self.pointer + 1) % self.legnth

    def _sampleFromHalfFull(self, batch_size):
        return self.buffer[randint(self.pointer, size=(batch_size))]

    def _sampleFromFull(self, batch_size):
        return self.buffer[randint(self.legnth, size=(batch_size))]

    def isFull(self):
        return self.filled

    def __getitem__(self, key):
        return self.buffer[:self._count, key]

    def __str__(self):
        return str(self.buffer)


class GeneralMemory(object):
    def __init__(self, size, *dimensions):
        self.memory = []
        for dim in dimensions:
            if type(dim) in [tuple, list]:
                self.memory.append(empty([size] + list(dim), dtype=float))
            elif dim == 0:
                self.memory.append(empty((size,), dtype=int))
            elif dim > 0:
                self.memory.append(empty((size, dim), dtype=float))
            else:
                self.memory.append(zeros((size,), dtype=bool))

        self.pointer = 0
        self.max_length = size
        self.length = 0
        self.add = self._fill
        self.filled = False
        self.added = 0

    @property
    def seed(self, value):
        seed(value)

    @property
    def isFull(self):
        return self.filled

    @property
    def count(self):
        if self.filled:
            return self.max_length
        else:
            return self.pointer

    def _fill(self, *inputs):
        i = self.pointer
        for b, m in enumerate(inputs):
            self.memory[b][i] = m
        self.pointer = (i + 1)
        self.length = (i + 1)
        if (i + 1) == self.max_length:
            self._switch()
        self.added += 1

    def addBatch(self, *inputs):
        length = len(inputs[0])
        self.length = min(self.max_length, self.pointer + length)
        self.added += length

        batch_start = 0
        pointer = self.pointer
        while length > 0:
            batch_length = min(length, self.max_length - pointer)
            for i, array in enumerate(inputs):
                self.memory[i][pointer:pointer +
                               batch_length] = array[batch_start:batch_start + batch_length]
            length -= batch_length
            batch_start += batch_length
            if (not self.filled) and (pointer + batch_length) >= self.max_length:
                self._switch()
            pointer = (pointer + batch_length) % self.max_length

        self.pointer = pointer

    def _switch(self):
        self.pointer = 0
        self.filled = True
        self.add = self._update

    def _update(self, *inputs):
        self.added += 1
        i = self.pointer
        for b, m in enumerate(inputs):
            self.memory[b][i] = m
        self.pointer = (i + 1) % self.max_length

    def getBatchIndex(self):
        return self.index

    def next(self):
        self.index = (self.index + 1) % self.length
        return [m[self.index] for m in self.memory]

    def sample(self, batch_size):
        self.index = randint(self.length, size=(batch_size))
        return [m[self.index] for m in self.memory]

    def __getitem__(self, key):
        return [m[key] for m in self.memory]

    def __setitem__(self, *args):
        key = args[0]
        for i, item in enumerate(args[1:][0]):
            self.memory[i][key] = copy(item)

    def __str__(self):
        return '\n'.join([str(m[:self.count()]) for m in self.memory])


class PrioritizedExperienceReplay(GeneralMemory):
    def __init__(self, alpha, size, *dimensions):
        self.alpha = alpha
        super(PrioritizedExperienceReplay, self).__init__(size, *dimensions)
        self.priority = zeros([size], dtype=float)

    def _fill(self, p, *inputs):
        i = self.pointer
        self.priority[i] = p
        for b, m in enumerate(inputs):
            self.memory[b][i] = m
        self.pointer = (i + 1)
        if (i + 1) == self.length:
            self._switch()

    def _update(self, p, *inputs):
        i = self.pointer
        self.priority[i] = p
        for b, m in enumerate(inputs):
            self.memory[b][i] = m
        self.pointer = (i + 1) % self.length

    def _sampleFromHalfFull(self, batch_size):
        try:
            self.index = choice(self.pointer, batch_size,
                                p=self.priority[:self.pointer] / sum(self.priority[:self.pointer]))
        except ValueError:
            print('vals', self.sum, sum(self.priority))
            raise
        return [m[self.index] for m in self.memory]

    def _sampleFromFull(self, batch_size):
        try:
            self.index = choice(self.length, batch_size,
                                p=self.priority / sum(self.priority))
        except ValueError:
            print('vals', self.sum, sum(self.priority))
            raise
        return [m[self.index] for m in self.memory]

    def getPriority(self):
        return self.priority[self.index]

    def updatePriority(self, priorities):
        self.priority[self.index] = power(priorities + 1e-7, .4)


if __name__ == "__main__":
    import numpy as np

    def healthy_copy():
        m = GeneralMemory(5, 1)
        m.count()
        m.addBatch(np.reshape(np.arange(5), (5, 1)))
        n = GeneralMemory(5, 1)
        n.add([0])
        n[0] = m[0]
        m[0][0][0] = 666
        return n[0][0][0] != m[0][0][0]

    def healthy_get_batch_index():
        m = GeneralMemory(5, 1)
        m.addBatch(np.reshape(np.arange(5), (5, 1)))
        m.sample(3)
        indexes = m.getBatchIndex()
        m.sample(5)
        return len(indexes) == 3

    def set_item():
        m = GeneralMemory(5, 1, 1)
        m[0] = [[1], [3]]
        a = (m.memory[0][0] == 1 and m.memory[1][0] == 3)
        m[0] = [1],[3]
        b = (m.memory[0][0] == 1 and m.memory[1][0] == 3)
        return a and b

    assert healthy_copy()
    assert healthy_get_batch_index()
    assert set_item()
