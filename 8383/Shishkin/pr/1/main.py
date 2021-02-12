def is_pyramid(n):
    if n == 0:
        return 0, False
    is_pyr = 1
    sum = 0
    k = 0
    while n != sum:
        if sum > n:
            is_pyr = 0
            break
        k += 1
        sum += k ** 2
    if is_pyr == 1:
        return k, True
    else:
        return k, False


n = input()
if n.isdigit():
    k, ans = is_pyramid(int(n))
    if ans:
        print(k)
    else:
        print("It is impossible")
else:
    print("Not a positive int number")
