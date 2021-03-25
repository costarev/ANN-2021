import numpy as np

arr = np.fromfile("input.csv", dtype=np.uint8, sep=" ")
res = np.unpackbits(arr[:, np.newaxis], axis=1)
np.savetxt("output.csv", res, delimiter=" ", fmt="%d")