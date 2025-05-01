import pandas as pd
import numpy as np
import random as rd

N = 10
d = 5
Xj = pd.DataFrame()

position = [[rd.uniform(0, 1) for j in range(d)]]

Xj['A'] = position
print(Xj)
