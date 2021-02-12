def omg(n):
    return (lambda x, length: sum(int(y) for y in x[:length // 2 if length % 2 != 0 else length // 2 - 1])
                              == sum(int(y) for y in x[length // 2 + 1:]))(str(n), len(str(n)))


print(omg(23441))
