# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:50:11 2020

@author: demon
"""

"""
测试一下各式采样，主要包括了
choices(population, weights=None, *, cum_weights=None, k=1) method of random.Random instance
Return a k sized list of population elements chosen with replacement.
——random.choices()


Choose a random element from a non-empty sequence.
——random.choice() 


sample(population, k) method of random.Random instance
Chooses k unique random elements from a population sequence or set.
    
Returns a new list containing elements from the population while
leaving the original population unchanged.
——random.sample()


This is an alias of `random_sample`. See `random_sample`  for the completedocumentation.
——np.random.sample() 


Generates a random sample from a given 1-D array
——np.random.choice() 
"""

import random
import numpy as np
from collections import deque


a = [[1,2,3,4,[5,6,7,8,],True,'a'],
     [1,2,3,4,[5,6,7,8,],False,'b'],
     [1,2,3,4,[5,6,7,8,],False,'c'],
     [1,2,3,4,[5,6,7,8,],True,'d'],
     [1,2,3,4,[5,6,7,8,],True,'e'],
     [1,2,3,4,[5,6,7,8,],True,'f'],
     [1,2,3,4,[5,6,7,8,],True,'g'],
     [1,2,3,4,[5,6,7,8,],True,'h'],
     [1,2,3,4,[5,6,7,8,],True,'i'],
     [1,2,3,4,[5,6,7,8,],True,'j'],
     [1,2,3,4,[5,6,7,8,],True,'k'],
     ]

b =deque(a)

c = np.array(a)

random.choice(a)
# [1, 2, 3, 4, [5, 6, 7, 8], True, 'h']



random.choices(a,k=15)
random.choices(b,k=15)
'''
[[1, 2, 3, 4, [5, 6, 7, 8], True, 'f'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'k'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'i'],
 [1, 2, 3, 4, [5, 6, 7, 8], False, 'b'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'k'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'f'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'h'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'e'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'f'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'h'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'a'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'i'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'e'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'h'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'g']]

[[1, 2, 3, 4, [5, 6, 7, 8], False, 'b'],
 [1, 2, 3, 4, [5, 6, 7, 8], False, 'c'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'j'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'k'],
 [1, 2, 3, 4, [5, 6, 7, 8], False, 'c'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'h'],
 [1, 2, 3, 4, [5, 6, 7, 8], False, 'c'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'a'],
 [1, 2, 3, 4, [5, 6, 7, 8], False, 'b'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'f'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'f'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'e'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'i'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'e'],
 [1, 2, 3, 4, [5, 6, 7, 8], True, 'g']]
'''

random.sample(a,k=15)
# Sample larger than population or is negative

np.random.choice(a)
np.random.choice(b)
np.random.choice(c)
# ValueError: a must be 1-dimensional

np.random.sample(a)
np.random.sample(b)
np.random.sample(c)
# TypeError: 'list' object cannot be interpreted as an integer
# TypeError: 'list' object cannot be interpreted as an integer
# TypeError: only integer scalar arrays can be converted to a scalar index




