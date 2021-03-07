from functools import reduce

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# np.random.seed(42)
sns.set_theme()

shape = np.fromfile("data/input.csv", dtype=int, sep=";")
assert all(map(lambda x: x > 0, shape)), "n m must be positive integers"

matrix = np.random.normal(0, 1, shape)
col_mean, col_variance = (lambda axis: (fn(matrix, axis=axis) for fn in (np.mean, np.var)))(0)

print(f"Generated matrix is:\n{matrix}\nMean vector:\n{col_mean}\nVariance vector:\n{col_variance}")
with open("data/output.txt", "w") as fout:
    fout.write(f"{matrix}\n{col_mean}\n{col_variance}")

count = reduce(lambda x, y: x * y, shape, 1)
rows = len(matrix)
df = pd.DataFrame({"row": matrix.reshape(count, ), "row_index": np.repeat(range(rows), count // rows)})

sns.displot(df, x="row", hue="row_index", binwidth=0.1, stat="density")
plt.savefig('data/hist.png')
plt.show()
