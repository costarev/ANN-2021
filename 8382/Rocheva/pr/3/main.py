import numpy as np
import matplotlib.pyplot as plt


def get_tensor(radius, color):
    t = []
    for x in range(0, radius*2):
        row = []
        for y in range(0, radius*2):
            if abs((x - radius) ** 2 + (y - radius) ** 2) <= radius ** 2:
                row.append(color)
            else:
                row.append([0, 0, 0])
        t.append(row)

    # print(type(t))
    tensor = np.array(t)
    # print(tensor.ndim, type(tensor))
    return tensor


def show_tensor(tensor):
    plt.imshow(tensor)
    # plt.show()
    plt.savefig('tensor.png')


data = np.fromfile('data.csv', dtype='int', sep=' ')
radius = data[0]
color = data[1:]

show_tensor(get_tensor(radius, color))