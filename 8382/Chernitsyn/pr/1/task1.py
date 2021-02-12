task1 = lambda s: sum([s.lower().count(c) for c in ('a', 'e', 'i', 'o', 'u')]) if type(s) is str else print("is not str")
print(task1("qwert"))