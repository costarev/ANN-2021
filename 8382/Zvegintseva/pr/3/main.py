import numpy as np
import matplotlib.pyplot as plt


def tensor_circle(radius, color):
    tensor = np.zeros((radius*2, radius*2, 3), dtype='int')

    for ind in np.ndindex(radius*2, radius*2):
        if abs((ind[0] - radius) ** 2 + (ind[1] - radius) ** 2) <= radius ** 2:
            tensor[ind[0]][ind[1]] = color

    return tensor


data_file = np.fromfile('data_circle.csv', dtype='int', sep=',')
radius = data_file[0]
color = data_file[1:4]

tensor = tensor_circle(radius, color)
plt.imshow(tensor)
plt.savefig('circle.png')