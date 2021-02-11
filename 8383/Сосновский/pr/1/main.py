def canBuildPyramide(n):
    ans = 0
    i = 1
    while True:
        ans += i ** 2
        if ans == n:
            return i
        elif ans > n:
            return "It is impossible"
        i += 1


print("Введите n:")
n = int(input())
print(canBuildPyramide(n))
