def product(n):
    #print(n)
    if n // 10 == 0:
        return 0
    res = 1
    while n > 0:
        digit = n % 10
        res = res * digit
        n = n // 10
    return res


def proof(num):
    count = 0
    current = product(num)
    while (current != 0):
        count += 1
        current = product(current)
    return count


if __name__ == "__main__":
    num = int(input("Number = "))
    print("Count ==", proof(num), sep=" ")
    input()

