import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 100, 1000)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xscale("log")

plt.show()
