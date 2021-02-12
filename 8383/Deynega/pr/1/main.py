import collections
from math import sqrt

def find_all(data):
    length = len(data)
    data = collections.Counter(data)
    Dx = 0
    Ex = 0
    for i in data:
        Dx += i**2 * data[i]
        Ex += i * data[i]
    Ex = Ex/length
    Dx = Dx/length - Ex**2
    return (Ex, sqrt(Dx))

#input_data = [0.1, 0.1, 0.4, 0.1, 0.4, 0.3, 0.1]
input_data = list((input("Type numbers splited by space:\n").split()))
input_data = [float(i) for i in input_data]
print(find_all(input_data))
