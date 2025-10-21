import math
import random

def inv_sqrt(x):
    y = (math.exp(-(x/2 + 0.2)) * 2.2 + 0.2) - x / 1024
    # print(y)
    for _ in range(5):
            y = y * (3 - x * y * y) / 2
    return y

for _ in range(100):
    x_tmp = random.uniform(0, 1) 
    print(x_tmp, inv_sqrt(x_tmp), 1/math.sqrt(x_tmp))