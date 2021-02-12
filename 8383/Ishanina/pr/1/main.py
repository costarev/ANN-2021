def decomposition_number(n):
    first = 0
    ans = ''
    i = 2
    count = 0
    while i * i <= n:
        while n % i == 0:
            if first == 0:
                ans += '(' + str(i)
                first = 1
            n = n / i
            count += 1
        if count == 1:
            ans += ')'
        elif count > 0:
            ans += '**' + str(count) + ')'
        i = i + 1
        count = 0
        first = 0
    if n > 1:
        ans += '(' + str(n) + ')'
    return ans


if __name__ == '__main__':
    num = int(input("Введите целое положительное число: "))
    if num <= 0:
        print('Было введено не целое положительное число!')
    else:
        answer = decomposition_number(num)
        print(answer)


