import sys
import numpy as np
import random
import pickle

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

dic = {'age': 3, 'name':'ming'}
dic_str = pickle.dumps(dic)
dic2 = pickle.loads(dic_str)
dic2['name'] = 'tom'
dic3 = pickle.loads(dic_str)
print(dic2 == dic3)
print(dic2)

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

class A(object):
    def __init__(self):
        self.name = 'tom'

class C(object):
    def __init__(self, objca):
        self.objca = objca

class B(object):
    def __init__(self, objba):
        self.objba = objba
        self.objc = C(self.objba)

def doSome():
        a = A()
        b = B(a)
        aa = A()
        aa.name = 'june'
        b.objba = aa
        b.objc = C(b.objba)
        print(b.objc.objca.name)

doSome()

