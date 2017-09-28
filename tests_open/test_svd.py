import time
import sys
import os
import numpy as np
import pandas as pd
import logging

print(sys.path)

from h2o4gpu.solvers.svd import SVDH2O

logging.basicConfig(level=logging.DEBUG)

def func():
    svd = SVDH2O(verbose=50)
    A = np.array([[1, 2, 0], [0, 0, 3], [1, 0, 4]])
    svd.fit(matrix=A)

def test_svd(): func()

