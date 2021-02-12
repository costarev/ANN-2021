#Задача № 10 - Максимова А.А.
def balance(numb):
    numb = str(numb)
    len_ = len(numb)
    flag = 0

    if len_ < 3:
        print("Число " + numb + " сбалансированное, так как 0 = 0.")
    else:
        lft = numb[:len_ // 2 - ((len_ - 1) % 2)]
        rgt = numb[len_ // 2 + 1:]
        if lft == rgt:
            print("Число " + numb + " сбалансированное, так как " + ' + '.join(list(lft)) + " = " + ' + '.join(list(rgt)) + ".")
        else:
            for i in range(0, len(lft)):
                flag = flag + int(lft[i]) - int(rgt[i])

            if not flag:
                print("Число " + numb + " сбалансированное, так как " + ' + '.join(list(lft)) + " = " + ' + '.join(list(rgt)) + ".")
            else:
                print("Число " + numb + " несбалансированное, так как " + ' + '.join(list(lft)) + " != " + ' + '.join(list(rgt)) + ".")


flag = True
while flag:
    print("Введите положительное целое число.")
    numb = input()
    if numb.isdigit():
        numb = int(numb)
        if numb > 0:
            flag = False
            balance(numb)
        else:
            print("Введен некорректный параметр. Попробуйте снова.")
    else:
        print("Введен некорректный параметр. Попробуйте снова.")

