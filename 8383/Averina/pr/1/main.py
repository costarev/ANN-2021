def task1(num):
    left = sum([int(x) for x in num[:int(len(num)/2)]])
    right = sum([int(x) for x in num[int(len(num)/2)+1:]])

    if right == left:
        print(str(left) + ' = ' + str(right))
        print('Число сбалансированное!')
    else:
        print(str(left) + ' != ' + str(right))
        print('Число несбалансированное!')


if __name__ == '__main__':
    task1(input('Введите число: '))

