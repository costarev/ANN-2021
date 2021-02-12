def task2(string):
    return len(set(string)) == len(string)


if __name__ == '__main__':
    answer = task2(input('Введите строку: '))

    if answer:
        print("Каждый символ в строке встречается только 1 раз")
    else:
        print("Есть повторяющиеся символы")