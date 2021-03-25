import numpy as np
x = np.genfromtxt('data1.csv', dtype=np.uint8, delimiter=' ')
np.savetxt('result1.csv', np.unpackbits(x[:, np.newaxis], axis=1), delimiter=' ', fmt='%i')
