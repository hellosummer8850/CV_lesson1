import numpy as np
import sklearn.datasets as Datasets
import seaborn as sn
import matplotlib.pyplot as plt

X,y = Datasets.make_friedman2(200, 0.3)
index = np.random.choice(range(len(X)), 3)
center = X[index, :]
print(center)