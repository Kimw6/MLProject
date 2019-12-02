import random


def balance_dt(data, label, seed=None):
    random.seed(seed)
    ones = []
    for i in range(len(label)):
        if label[i] == 1:
            ones.append(i)
    zeros = []
    for i in range(len(label)):
        if label[i] == 0:
            zeros.append(i)
    zeros = random.sample(zeros, len(ones))
    indices = zeros + ones
    X = data[indices]
    y = label[indices]
    return X, y


def balance_reversed(data, label, seed=None):
    random.seed(seed)
    zeros = []
    for i in range(len(label)):
        if label[i] == 0:
            zeros.append(i)
    ones = []
    for i in range(len(label)):
        if label[i] == 1:
            ones.append(i)
    ones = random.sample(ones, len(zeros))
    indices = zeros + ones
    X = data[indices]
    y = label[indices]
    return X, y