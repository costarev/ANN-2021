import numpy as np
import math
import matplotlib.pyplot as plt


def circle(radius, color):
    height = width = radius * 2
    canvas = np.zeros([height, width, 3], dtype='int')

    y_center = (height - 1) / 2
    x_center = (width - 1) / 2

    for y in range(height):
        for x in range(width):
            if math.sqrt((y_center - y)**2 + (x_center - x)**2) <= radius:
                canvas[y][x] = color

    return canvas


file_input = np.fromfile('input.csv', dtype='int', sep=';')
radius = file_input[0]
color = file_input[1:4]

canvas = circle(radius, color)
# canvas.tofile('output.csv', ';', '%d')

plt.imshow(canvas)
plt.savefig('circle.png')
