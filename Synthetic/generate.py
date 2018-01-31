import ujson as json
import numpy as np
import matplotlib.pyplot as plt

SEQ_LEN = 36

def gen_item(rate = 0.05):
    p = np.random.randint(2, 8)
    buffer_ = np.random.random(p - 1).tolist()

    zeta = 0.0
    mu = 0.0
    x = []

    for t in range(SEQ_LEN + 1):
        eta = -np.sum(buffer_[-(p - 1):]) + np.random.randn() * 0.3
        buffer_.append(eta)

        mu += zeta + np.random.randn() * 0.3
        zeta += np.random.randn() * 0.3
        x.append(mu + eta + np.random.randn() * 0.3)

    x, y = x[:-1], x[-1]
    x = x[::-1]

    mask = [True] * len(x)

    for t in range(SEQ_LEN):
        if np.random.random() < rate:
            for k in range(5):
                if t + k < SEQ_LEN:
                    mask[t + k] = False

    return x, y, mask
