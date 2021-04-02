import matplotlib.pyplot as plt
import numpy as np
import torch
import timeit


def create_tensor_v_2(data):
    size = 2*data[0]+50
    tensor = torch.randint(0, 1, (size, size, 3))
    colour = torch.tensor([data[1], data[2], data[3]])
    nrows, ncols = size, size
    row, col = np.ogrid[:nrows, :ncols]
    cnt_row, cnt_col = nrows / 2, ncols / 2
    mask = ((row - cnt_row) ** 2 + (col - cnt_col) ** 2 < data[0]**2)
    tensor[mask] = colour
    return tensor


def create_tensor_v_1(data):
    size = 2*data[0]+50
    tensor = torch.randint(0, 1, (size, size, 3))
    colour = torch.tensor([data[1], data[2], data[3]])
    for i in range(size):
        for j in range(size):
            if ((j - size // 2) ** 2 + (i - size // 2) ** 2) < data[0] ** 2:
                tensor[i][j] = colour
    return tensor


#input_data = [50, 255, 0, 125]

input_data = np.fromfile('input_file.txt', dtype=int, sep=' ')

# start_time = timeit.default_timer()
# tensor = create_tensor_v_2(input_data.tolist())
# print(timeit.default_timer() - start_time)

tensor = create_tensor_v_2(input_data.tolist())

np.save('bin_tensor', tensor)

#tensor = np.load('bin_tensor.npy')

plt.imshow(tensor)
plt.axis('off')
plt.show()
