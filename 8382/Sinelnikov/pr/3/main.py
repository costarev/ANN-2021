import numpy as np

try:
    data = np.loadtxt("data.txt")

    mean = np.mean(data)

    std = np.std(data)

    standart_data = (data - mean) / std

    np.savetxt("output.txt", standart_data)

except:
  print("Данные введены некорректно")


