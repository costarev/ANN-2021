def multiply_digits(number):
    result = 1
    while number != 0:
        result = result * (number % 10)
        number = number // 10
    return result


def count_multiplications(number):
    if number < 0:
        return -1
    result = 0
    while number > 10:
        number = multiply_digits(number)
        result += 1
    return result


num = int(input())
print(count_multiplications(num))
