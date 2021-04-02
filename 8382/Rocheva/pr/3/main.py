import numpy as np
import matplotlib.pyplot as plt


def get_tensor(radius, color):
    tensor = np.zeros((radius*2, radius*2, 3), dtype='int')

    for x in range(0, radius*2):
        for y in range(0, radius*2):
            if abs((x - radius) ** 2 + (y - radius) ** 2) <= radius ** 2:
                tensor[x][y] = color

    return tensor


def show_tensor(tensor):
    plt.imshow(tensor)
    # plt.show()
    plt.savefig('tensor.png')


data = np.fromfile('data.csv', dtype='int', sep=' ')
radius = data[0]
color = data[1:]

show_tensor(get_tensor(radius, color))
