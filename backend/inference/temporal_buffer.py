from collections import deque
import numpy as np

class TemporalBuffer:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)

    def add(self, feature_vector):
        self.buffer.append(feature_vector)

    def is_ready(self):
        return len(self.buffer) == self.sequence_length

    def get_sequence(self):
        return np.array(self.buffer, dtype=float)
