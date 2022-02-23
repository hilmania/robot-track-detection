import numpy as np
from matplotlib import pyplot as plt


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    # return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return np.rad2deg(ang1 - ang2)

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
data = np.array([[0, 1], [0, -1]])
origin = np.array([[0, 0], [0, 0]])
plt.quiver(*origin, data[:, 0], data[:, 1], color=['black', 'red', 'green'], scale=5)

A = (0, 1)
B = (0, -1)

print(angle_between(A, B))

# 45.

print(angle_between(B, A))

# 315.

plt.show()
