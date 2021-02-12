try:
    a = int(input("Enter number: "))

    if (a < 2):
        print(a)
    numbers = []

    for i in range(2, a + 1):
        count = 0

        while a % i == 0:
            a = a // i
            count += 1

        if count > 0:
            numbers.append("({0}**{1})".format(i, count))

        if a == 1:
            break

    print("".join(numbers))
except:
    print("Hey, are you sure you entered just integer number? You killed me...")