def balanced(num):
    if (num <= 0):
        raise ValueError("A positive number should be entered")
    str_num = str(num)
    n = (len(str_num) - 1)//2
    sum_left = 0
    sum_right = 0
    for i in range(0, n):
        sum_left += int(str_num[i])
        sum_right += int(str_num[-i - 1])
    return sum_left == sum_right

try:
    is_balanced = balanced(int(input("Enter a number: ")))
    if is_balanced:
        print("entered number is balanced")
    else:
        print("entered number is NOT balanced")
except ValueError:
    print("Incorrect input")
