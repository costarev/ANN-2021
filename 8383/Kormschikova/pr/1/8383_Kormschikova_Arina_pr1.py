def balance(n):
    if n < 0:
        print("Wrong input!")
        return
    list_num = list(str(n))
    l = len(list_num)
    if l%2 == 0:
        l-=2
    else:
        l-=1
    l//=2

    sumLeft = 0 
    sumRight = 0  
    i_ = -1
    for i in range(0,l):
        sumLeft+=int(list_num[i])
        sumRight+=int(list_num[i_])
        i_-=1
    if sumLeft == sumRight:
        print("Число сбалансированное", sumLeft, "=", sumRight)
    else:
        print("Число несбалансированное", sumLeft, "!=", sumRight)


n = int(input('Введите n: '))
balance(n)
