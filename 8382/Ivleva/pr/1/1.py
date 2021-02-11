def result(number):
    res = ""
    power = 0
    i = 2
    while number != 1:
        while (number % i == 0):
            power += 1
            number = number // i
        if power > 1:
            res = res + "(" + str(i) + "**" + str(power) + ")"
        elif power == 1:
            res = res + "(" + str(i) + ")"
        i += 1
        power = 0
    return res


number = int(input("Введите положительное число: "))
while number <= 0:
    number = int(input("Введите положительное число: "))
print(result(number))
