def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def simple(n):
     dec = ""
     f = n
     for i in range(2, isqrt(n)):
         degree = 0
         while n % i == 0:
             n //= i
             degree += 1
         if degree > 1:
             dec = dec + "(" + str(i) + "**" + str(degree) +")"
         elif degree == 1:
             dec = dec + "(" + str(i) +")"

     if (n != 1):
         if (n != f):
            dec = dec + "(" + str(n) + ")"
         else:
            dec = dec + str(f)

     return dec


try:
    n = int(input("Введите целое положительное число: "))
    print("Его разложение на множители: " + simple(n)) if n > 0 else print("Неверный ввод")

except:
    print("Неверный ввод")
