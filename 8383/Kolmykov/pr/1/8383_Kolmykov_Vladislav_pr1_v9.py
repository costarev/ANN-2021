def f(n):
    if n <= 0:
        return "It is impossible"
    k = 0
    num = 0
    while num < n:
        k += 1
        num += k ** 2
        if num > n:
            return "It is impossible"
        elif num == n:
            return k


print(f(1))
print(f(5))
print(f(14))
print(f(30))

print(f(0))
print(f(3))
print(f(10))
print(f(20))
