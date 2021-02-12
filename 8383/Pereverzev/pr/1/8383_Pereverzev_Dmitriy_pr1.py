def func_1(number):
    step = 0
    pre_num = 0
    if number <= 0:
        return 'It is impossible'
    while number > pre_num:
        step = step + 1
        pre_num = pre_num + step * step
        if number == pre_num:
            return step
        elif number < pre_num:
            return 'It is impossible'


while(True):
    inp = input()
    if(inp.isdigit()):
        print(func_1(int(inp)))
    elif(inp == 'stop'):
        exit()
    else:
        print("need int(>0)||'stop'")

