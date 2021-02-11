def isBalance(num):
    x = str(num)
    length = len(x)
    if length % 2 != 0:
        start = sum(int(y) for y in x[:length // 2])
    else:
        start = sum(int(y) for y in x[:length // 2 - 1])

    end = sum(int(y) for y in x[length // 2 + 1:])
    return start == end


n = input("Введите n: ")
try:
    n = int(n)
    if isBalance(n):
        print("Число " + str(n) + " является сбалансированным")
    else:
        print("Число " + str(n) + " не является сбалансированным")
except ValueError:
    print("Введите целое число!")


