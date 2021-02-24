import numpy as np
import matplotlib.pyplot as plt


def expectation(a, n):
    prb = 1 / n
    sum = 0
    for i in range(0, n):
        sum += (a[i] * prb)
    return float(sum)


M = input()
if not M.isdigit():
    print("M is not a positive int number")
    raise SystemExit

N = input()
if not N.isdigit():
    print("N is not a positive int number")
    raise SystemExit

M = int(M)
N = int(N)

matr = np.random.normal(0, 1, size=(M, N))
print("Generated matrix:")
print(matr)

expectations = np.zeros(N)
vars = np.zeros(N)
i = 0
b = matr.T
for cell in b:
    expectations[i] = expectation(cell, M)
    vars[i] = np.var(cell)
    i += 1

print("Expected value:", end=' ')
print(expectations)
print("Variance:", end=' ')
print(vars)

plt.hist(matr)
plt.xlabel("Value")
plt.ylabel("Label")
plt.title("Hist for each line")
plt.show()
