import numpy as np
from collections import deque


class Buffer(object):

    def __init__(self, length, shape):
        self.buffer = np.empty([length] + list(shape))
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
        return self.buffer[np.random.randint(self.pointer, size=(batch_size))]

    def _sampleFromFull(self, batch_size):
        return self.buffer[np.random.randint(self.legnth, size=(batch_size))]

    def isFull(self):
        return self.filled

    def __getitem__(self, key):
        return self.buffer[:self._count, key]

    def __str__(self):
        return str(self.buffer)


class Memory(object):

    def __init__(self, size, *dimensions):
        assert type(size) == int, "Argument `size` should have type `int`. %s, %s" % (
            size, type(size))
        self.memory = []
        for dim, t in dimensions:
            if type(dim) in [tuple, list]:
                m = np.zeros([size] + list(dim), dtype=t)
                self.memory.append(m)
            elif dim == 0:
                self.memory.append(np.zeros((size,), dtype=t))
            elif dim > 0:
                self.memory.append(np.zeros((size, int(dim)), dtype=t))
            else:
                raise ValueError

        self.pointer = 0
        self.max_length = size
        self.length = 0
        self.add = self._fill
        self.filled = False
        self.added = 0

    def save(self, path):
        import os
        if os.path.exists(path + '.npz'):
            raise FileExistsError("file with this name already exists")
        self.fit()
        dictionary = {'%i' % i: self.memory[i]
                      for i in range(len(self.memory))}
        np.savez(path, **dictionary)

    def fit(self):
        l = len(self)
        for i in range(len(self.memory)):
            self.memory[i] = self.memory[i][:l]

    def load(self, path):
        if path[-4:] != '.npz':
            path += '.npz'
        dictionary = np.load(path)

        assert self.max_length >= dictionary['0'].shape[0]

        self.addBatch(*[dictionary['%i' % i]
                        for i in range(len(dictionary.keys()))])

    def seed(self, value):
        np.seed(value)

    def isFull(self):
        return self.filled

    def shape(self):
        return [m.shape for m in self.memory]

    def size(self):
        return self.max_length

    def __len__(self):
        if self.isFull():
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

    def _update(self, *inputs):
        self.added += 1
        i = self.pointer
        for b, m in enumerate(inputs):
            self.memory[b][i] = m
        self.pointer = (i + 1) % self.max_length

    def _switch(self):
        self.pointer = 0
        self.filled = True
        self.add = self._update

    def addBatch(self, *inputs):
        length = len(inputs[0])
        self.length = max(self.length, min(
            self.max_length, self.pointer + length))
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

    def getBatchIndex(self):
        return self.index

    def next(self):
        # maybe should use the method instead
        self.index = (self.index + 1) % self.length
        return [m[self.index] for m in self.memory]

    def sample(self, batch_size):
        self.index = np.random.randint(self.length, size=(batch_size))
        return [m[self.index] for m in self.memory]

    def __getitem__(self, key):
        return [m[key] for m in self.memory]

    def __setitem__(self, *args):
        key = args[0]
        for i, item in enumerate(args[1:][0]):
            self.memory[i][key] = np.copy(item)

    def __str__(self):
        size = self.__len__()
        return '\n'.join([str(m[:size]) for m in self.memory])

    def nbytes(self, unit='b'):
        unit = unit.lower()
        if unit in ['b', 'byte', 'bytes']:
            c = 1
        elif unit in ['k', 'kb', 'kilo', 'kilobytes']:
            c = 1024
        elif unit in ['m', 'mb', 'mega', 'megabytes']:
            c = 1024**2
        elif unit in ['g', 'gb', 'giga', 'gigabytes']:
            c = 1024**3
        else:
            raise ValueError

        return np.sum([a.nbytes for a in self.memory]) / c


class ContinuousMemory(Memory):

    def __init__(self, memory_size, state_structure, action_structure, reward_type=np.float32):
        super().__init__(
            memory_size,
            (0, np.int32),
            state_structure,
            action_structure,
            (0, reward_type),
            (0, np.bool),
        )
        self.arg_count = 4  # TODO: support nested state structure
        self.exp_counter = 0

    def _fill(self, *inputs):
        if len(inputs) < self.arg_count:
            self.cut()
        i = self.pointer
        for b, m in enumerate([self.exp_counter, *inputs]):
            self.memory[b][i] = m
        self.pointer = (i + 1)
        self.length = (i + 1)
        if (i + 1) == self.max_length:
            self._switch()
        self.added += 1
        self.exp_counter += 1

    def _update(self, *inputs):
        if len(inputs) < self.arg_count:
            self.cut()
        self.added += 1
        i = self.pointer
        for b, m in enumerate([self.exp_counter, *inputs]):
            self.memory[b][i] = m
        self.pointer = (i + 1) % self.max_length
        self.exp_counter += 1

    def cut(self):
        self.exp_counter += 1

    def sample(self, batch_size, length=1):
        exps = [super().sample(batch_size)]
        exps += [self.next() for _ in range(length)]
        seq = exps[-1][0] - exps[0][0] == length
        # print(exps)

        state = [e[1][seq] for e in exps]
        action = exps[-1][-3][seq]
        reward = exps[-1][-2][seq]
        terminal = exps[-1][-1][seq]

        return state, action, reward, terminal


class EpisodicMemory(object):

    def __init__(self, memory_size):
        self.episodes = []
        self.count = 0
        self.episodes_size = memory_size

    def add(self, trajectory, lenght=0):
        self.episodes.append(trajectory)
        self.count += (length or len(trajectory))
        if self.count > self.episodes_size:
            self.episodes = self.episodes[1:]

    def sample(self):
        return self.episodes[np.random.randint(len(self.episodes))]


class PrioritizedExperienceReplay(Memory):

    def __init__(self, alpha, size, *dimensions):
        self.alpha = alpha
        super(PrioritizedExperienceReplay, self).__init__(size, *dimensions)
        self.priority = np.zeros([size], dtype=float)

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
            p = self.priority[:self.pointer] / \
                np.sum(self.priority[:self.pointer])
            self.index = np.choice(self.pointer, batch_size, p=p)
        except ValueError:
            print('vals', self.sum, np.sum(self.priority))
            raise
        return [m[self.index] for m in self.memory]

    def _sampleFromFull(self, batch_size):
        try:
            self.index = np.choice(self.length, batch_size,
                                   p=self.priority / np.sum(self.priority))
        except ValueError:
            print('vals', self.sum, np.sum(self.priority))
            raise
        return [m[self.index] for m in self.memory]

    def getPriority(self):
        return self.priority[self.index]

    def updatePriority(self, priorities):
        self.priority[self.index] = np.power(priorities + 1e-7, .4)


if __name__ == "__main__":
    import numpy as np

    def healthy_copy():
        m = Memory(5, (1, np.int32))
        m.addBatch(np.reshape(np.arange(5), (5, 1)))
        n = Memory(5, (1, np.int32))
        n.add([0])
        n[0] = m[0]
        m[0][0][0] = 666
        return n[0][0][0] != m[0][0][0]

    def healthy_get_batch_index():
        m = Memory(5, (1, np.int32))
        m.addBatch(np.reshape(np.arange(5), (5, 1)))
        m.sample(3)
        indexes = m.getBatchIndex()
        m.sample(5)
        return len(indexes) == 3

    def set_item():
        m = Memory(5, (1, np.int32), (1, np.int32))
        m[0] = [[1], [3]]
        a = (m.memory[0][0] == 1 and m.memory[1][0] == 3)
        m[0] = [1], [3]
        b = (m.memory[0][0] == 1 and m.memory[1][0] == 3)
        return a and b

    def setter():
        m = Memory(5, (1, np.int32), (3, np.int32))
        m[1] = 1, [2, 2, 2]
        a = m.memory[0][1][0] == 1
        b = np.all([m.memory[1][1][i] == 2 for i in range(3)])
        return a and b

    def storeLoad():
        m = Memory(5, (1, np.int32), (3, np.int32))
        m.add(0, (1, 2, 3))
        m.add(4, (5, 6, 7))
        m.add(8, (9, 10, 11))
        print(m)
        print('--------------')
        m.save('/tmp/memory_numpy_save_test')
        m.load('/tmp/memory_numpy_save_test')
        print(m)

    # assert healthy_copy()
    # assert healthy_get_batch_index()
    # assert set_item()
    # assert setter()
    # storeLoad()

    m = ContinuousMemory(10, (2, np.int8), (0, np.float))
    m.add([3, 4])
    m.add([2, 3], 3, 1, False)
    m.add([4, 5], 1, 1, True)
    print(m)
    print(m.sample(5))
