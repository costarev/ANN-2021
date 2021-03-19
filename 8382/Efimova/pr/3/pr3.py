import numpy as np

file = np.fromfile('/content/sample_data/data.csv', dtype='int', sep=';')

def Chess(arr):
  a = arr[0]
  b = arr[1]
  c = arr[2]
  d = arr[3]
  res = np.ones([a, b])
  res[0:a:2, 0:b:2] = c
  res[0:a:2, 1:b:2] = d
  res[1:a:2, 0:b:2] = d
  res[1:a:2, 1:b:2] = c
  return res

result = Chess(file)

np.savetxt('result_new.csv', result, fmt="%d", delimiter=' ')
np.savetxt('result_new1.csv', result, fmt="%d", delimiter=',')
