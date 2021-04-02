def pyramide(n):
    s = 0
    for i in range(1, 10000):
        s += i**2
        if s == n:
            print(i)
            break
        if s > n:
            print("It's impossible")
            break


n = int(input("Enter number: "))
pyramide(n)
