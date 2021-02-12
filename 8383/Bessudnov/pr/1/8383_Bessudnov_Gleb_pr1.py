def f(number):
    ##Изначально принимаю, что в числе нет единичных битов
    answer = 0
    ##Перебираю цифры чила, пока не дойду до 0
    while number:
        ##Если конечный бит числа равен 1, то он прибавится, если нет, то прибавится 0
        answer += number & 1
        ##Делаю сдвиг вправо, чтобы просмотреть следующую цифру числа
        number = number >> 1
    return answer   

##Ввод числа
n = int(input("Enter number: "))
##Вызов функции из задания
bitsCount = f(n)
##Печатаю ответ
print("1 bits count in number", n, "is ", bitsCount, "(", bin(n), ")")