import numpy as np

index = np.arange(10)
np.random.shuffle(index)
a = np.arange(10)*2

print(a[index])
