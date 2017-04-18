from numpy import empty, zeros
from numpy.random import randint, seed


class Memory(object):
    def __init__(self, size, state_dims, action_dims):
        if type(state_dims) == int:
            state_dims = [state_dims]
        self.states = empty([size] + state_dims, dtype=float)

        if action_dims == 0:
            self.actions = empty(size, dtype=int)
        else:
            self.actions = empty((size, action_dims), dtype=float)
        self.rewards = empty((size, 1), dtype=float)
        self.states_next = empty(([size] + state_dims), dtype=float)
        self.terminations = zeros((size, 1), dtype=bool)
        self.memory = [self.states, self.actions,
                       self.rewards, self.states_next, self.terminations]
        self.pointer = 0
        self.length = size
        self.add = self._fill
        self.sample = self._sampleFromHalfFull
        self.filled = False

    def setSeed(self, value):
        seed(value)

    def isFull(self):
        return self.filled

    def count(self):
        if self.filled:
            return self.length
        else:
            return self.pointer

    def _fill(self, state, action, reward, next_state, done=0):
        i = self.pointer
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.states_next[i] = next_state
        self.terminations[i] = done
        self.pointer = (i + 1)
        if (i + 1) == self.length:
            self._switch()

    def _switch(self):
        self.pointer = 0
        self.filled = True
        self.add = self._update
        self.sample = self._sampleFromFull

    def _update(self, state, action, reward, next_state, done=0):
        i = self.pointer
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.states_next[i] = next_state
        self.terminations[i] = done
        self.pointer = (i + 1) % self.length

    def _sampleFromHalfFull(self, batch_size):
        index = randint(self.pointer, size=(batch_size))
        return self.states[index], self.actions[index], self.rewards[index], self.states_next[index], self.terminations[index]

    def _sampleFromFull(self, batch_size):
        index = randint(self.length, size=(batch_size))
        return self.states[index], self.actions[index], self.rewards[index], self.states_next[index], self.terminations[index]

    def getAll(self):
        return self.states[:self.length], self.actions[:self.length], self.rewards[:self.length], self.states_next[:self.length], self.terminations[:self.length]

    def addBatch(self, states, actions, rewards, next_states, terminations=None, length=None):
        length = length or len(states)
        memory_length = self.length

        batch_start = 0
        pointer = self.pointer
        while length > 0:
            batch_length = min(length, memory_length - pointer)
            self.states[
                pointer:pointer + batch_length] = states[batch_start:batch_start + batch_length]
            self.actions[
                pointer:pointer + batch_length] = actions[batch_start:batch_start + batch_length]
            self.rewards[
                pointer:pointer + batch_length] = rewards[batch_start:batch_start + batch_length]
            self.states_next[
                pointer:pointer + batch_length] = next_states[batch_start:batch_start + batch_length]
            if terminations:
                self.terminations[
                    pointer:pointer + batch_length] = terminations[batch_start:batch_start + batch_length]
            length -= batch_length
            batch_start += batch_length
            if (not self.filled) and (pointer + batch_length) >= memory_length:
                self._switch()
            pointer = (pointer + batch_length) % memory_length

        self.pointer = pointer

    def merge(self, other):
        assert type(other) == type(self)
        if other.filled:
            self.addBatch(other.states, other.actions, other.rewards,
                          other.states_next, other.terminations)
        else:
            self.addBatch(other.states, other.actions, other.rewards,
                          other.states_next, other.terminations, other.pointer)


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
    def __init__(self, size, dimensions):
        self.memory = []
        for dim in dimensions:
            if dim == 0:
                self.memory.append(empty((size,), dtype=int))
            else:
                self.memory.append(empty((size, dim), dtype=float))

        self.pointer = 0
        self.length = size
        self.add = self._fill
        self.sample = self._sampleFromHalfFull
        self.filled = False

    def setSeed(self, value):
        seed(value)

    def isFull(self):
        return self.filled

    def count(self):
        if self.filled:
            return self.length
        else:
            return self.pointer

    def _fill(self, inputs):
        i = self.pointer
        for b, m in enumerate(self.memory):
            m[i] = inputs[b]
        self.pointer = (i + 1)
        if (i + 1) == self.length:
            self._switch()

    def _switch(self):
        self.pointer = 0
        self.filled = True
        self.add = self._update
        self.sample = self._sampleFromFull

    def _update(self, inputs):
        i = self.pointer
        for b, m in enumerate(self.memory):
            m[i] = inputs[b]
        self.pointer = (i + 1) % self.length

    def _sampleFromHalfFull(self, batch_size):
        index = randint(self.pointer, size=(batch_size))
        return [m[index] for m in self.memory]

    def _sampleFromFull(self, batch_size):
        index = randint(self.length, size=(batch_size))
        return [m[index] for m in self.memory]


if __name__ == "__main__":
    m = GeneralMemory(10, [1, 2, 0])
    m.add([[3], [45, 6], 1.5])

    print(m.sample(5))
