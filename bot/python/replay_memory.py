import random
import math
from collections import namedtuple
from torch.utils.data import Dataset

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'next_action', 'reward', 'deltaTime'))


class ReplayMemory(Dataset):
    def __init__(self, capacity, prioritized_replay):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.max_error = 8
        self.prioritized_replay = prioritized_replay

        if self.prioritized_replay:
            self.error_buckets = []
            self.priority_samples = []
            for i in range(10):
                self.error_buckets.append([])

            self.count = 0

    def push(self, transition: Transition):
        """Saves a transition."""
        if self.prioritized_replay:
            random.choice(self.error_buckets).append(transition)
            self.count += 1
        else:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.prioritized_replay:
            result = []
            while len(result) < batch_size and len(self.priority_samples) > 0:
                result.append(self.priority_samples.pop())

            changed = True
            while len(result) < batch_size and changed:
                changed = False
                for bucket in self.error_buckets:
                    # Take N samples from each bucket according to the log2 of their size.
                    n = min(len(bucket), int(math.log2(max(2, len(bucket)))))
                    for i in range(n):
                        changed = True
                        idx = random.randrange(0, len(bucket))
                        result.append(bucket[idx])
                        bucket[idx] = bucket[-1]
                        bucket.pop()
                        self.count -= 1
            return result
        else:
            return random.sample(self.memory, batch_size)

    def get_all(self):
        if self.prioritized_replay:
            raise NotImplementedError()
        else:
            return self.memory

    def discard_random(self):
        if not self.prioritized_replay:
            assert False
        idx = random.randrange(0, self.count)
        for bucket in self.error_buckets:
            if idx < len(bucket):
                bucket[idx] = bucket[-1]
                bucket.pop()
                self.count -= 1
                return
            idx -= len(bucket)

        assert False, (idx, self.count, sum(map(len, self.error_buckets)))

    def insert(self, samples, errors):
        if not self.prioritized_replay:
            return

        assert len(samples) == len(errors)
        for i in range(len(samples)):
            # Discard a random sample
            while self.count >= self.capacity:
                self.discard_random()

            bucket_idx = max(0, min(len(self.error_buckets) - 1, math.floor(len(self.error_buckets) * errors[i].item() / self.max_error)))
            self.error_buckets[bucket_idx].append(samples[i])
            self.count += 1

    def __len__(self):
        return self.count if self.prioritized_replay else len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]
