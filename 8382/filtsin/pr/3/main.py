import numpy as np

np.savetxt("out.txt", np.unpackbits(np.array(np.fromfile("file.txt", sep=',', dtype=np.uint64)[:, np.newaxis], dtype='>u8').view(np.uint8), axis=1), fmt="%d")