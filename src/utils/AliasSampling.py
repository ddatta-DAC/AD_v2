import numpy as np
import os
import sys
import random
from collections import Counter


class alais_sampling_obj():
    def __init__(self, list_prob):
        self.list_prob = list_prob
        self.setup()
        return

    def setup(self):
        self.n = len(self.list_prob)
        probs = self.list_prob

        n = len(probs)
        q = np.zeros(n)
        J = np.zeros(n, dtype=np.int)

        smaller = []
        larger = []

        for _i, prob in enumerate(probs):
            q[_i] = n * prob
            if q[_i] < 1.0:
                smaller.append(_i)
            else:
                larger.append(_i)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1
            if q[large] < 1:
                smaller.append(large)
            else:
                larger.append(large)

        self.prob = q
        self.alias = J
        return

    # ===================================
    # Generate Single draw
    # According to set Discrete probability distribution
    # ===================================

    def generate(self):
        i = int(np.floor(np.random.rand() * self.n))
        p_i = np.random.rand()
        if p_i < self.prob[i]:
            return i
        else:
            return int(self.alias[i])

# ----------------------------------------------- #

def test():
    p_list = [1 / 4, 1 / 8, 1 / 8, 1 / 2]
    obj = alais_sampling(p_list)
    res = []
    for _ in range(10000):
        res.append(obj.generate())
    print(Counter(res))
