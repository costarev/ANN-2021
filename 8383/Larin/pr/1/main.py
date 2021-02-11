def task1(s:str):
    d = {}
    for i in 'aeiou': d[i]=0
    for c in s:
        if(c in 'aeiou'): d[c] +=1
    return d

if __name__ == "__main__":
    print(task1(input()))
