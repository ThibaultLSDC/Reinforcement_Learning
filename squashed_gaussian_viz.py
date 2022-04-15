import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(100000,) * 1.5 + .5

plt.hist(np.tanh(x), bins=200, density=True)
plt.show()
