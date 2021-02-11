def count_ones(num):
    n = num
    s = ""
    count = 0
    while num > 0:
        bit = num % 2
        count += bit
        s = str(bit) + s
        num //= 2
    print('Число ', n, ' - ', s, ' содержит ', count, 'единиц')
    return count


try:
    value = input('Введите число - ')
    value = int(value)
    count_ones(value)
except:
    print('Введено что-то неправильное!')

# count_ones(1)
# count_ones(2)
# count_ones(7)
# count_ones(16)
# count_ones(37)
# count_ones(4095)
