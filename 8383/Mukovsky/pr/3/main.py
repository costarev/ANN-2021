import numpy as np
import matplotlib.pyplot as plt
import math


def get_tensor(radius, color):
    size = radius * 2
    center = (size - 1) / 2
    tensor = np.zeros((size, size, 3), dtype='int')
    for ind in np.ndindex(size, size):
        if math.sqrt((center - ind[1]) ** 2 + (center - ind[0]) ** 2) <= radius:
            tensor[ind[0]][ind[1]] = color
    return tensor


data = np.fromfile('circle.csv', dtype='int', sep=';')

radius = data[0]
color = data[1:]

tensor = get_tensor(radius, color)
plt.imshow(tensor)
plt.savefig('result.png')
