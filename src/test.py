import sys
import numpy as np
import random

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

print(random.random()<0.3)

act_visits = []
if len(act_visits) == 0:
    act_visits.append((0, 50))
acts, visits = zip(*act_visits)
act_probs = softmax(1.0 / 1e-3 * np.log(visits))
print(acts, act_probs)

print(np.asarray([5])[:, None])
print(sys.path[0].index('src'))
print(sys.path[0].split('src')[0])