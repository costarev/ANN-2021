import numpy as np
import matplotlib.pyplot as plt

def f(M, N):
	file = open("out.txt", "w")

	matr = np.random.normal(size=[M, N])
	file.write('Массив:\n' + str(matr) + '\n')

	M = np.mean(matr, axis=0)
	file.write('Мат ожидание:\n' + str(M) + '\n')

	disp = np.var(matr, axis=0)
	file.write('Дисперсия:\n:' + str(disp) + '\n')
	
	for string in matr:
		plt.hist(string)
		plt.show()

M, N = np.loadtxt("in.txt", dtype=int)
f(M, N)