def rmsd(v1, v2):
    numerator = list(map(lambda x, y: (x - y) ** 2, v1, v2))
    return (sum(numerator) / (len(numerator))) ** 0.5


while True:
    try:
        vector_str1 = input('Input first vector:\n').split(' ')
        vector1 = list(map(int, vector_str1))
        vector_str2 = input('Input second vector:\n').split(' ')
        vector2 = list(map(int, vector_str2))
    except ValueError:
        print("Incorrect input, enter again")
        continue
    if len(vector1) != len(vector2):
        print('Enter vectors of the same length')
    else:
        print(f'Root-mean-square deviation = {rmsd(vector1, vector2)}')
        break
