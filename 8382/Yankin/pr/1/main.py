def steps_to_one_numeral(num):
    if type(num) is not int or num < 1:
        return -1

    count = 0
    while len(str(num)) > 1:
        numerals = list(str(num))
        num = 1
        for numeral in numerals:
            num *= int(numeral)
        count += 1
    return count


print("Шагов для 39: ", steps_to_one_numeral(39))
print("Шагов для 4: ", steps_to_one_numeral(4))
print("Шагов для 999: ", steps_to_one_numeral(999))
print("")


print("Количество перемножений цифр числа до получения числа из одной цифры")
print("Выход – exit")
print("")


user_input = ""
while user_input != "exit":
    user_input = input("Введите число: ")
    if not user_input.isdigit():
        continue
    print('Количество шагов:', steps_to_one_numeral(int(user_input)))
