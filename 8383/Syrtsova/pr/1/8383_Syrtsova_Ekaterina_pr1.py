s = input()
vowels = 0
for i in s:
    letter = i.lower()
    if letter == "a" or letter == "e" or letter == "i"\
            or letter == "o" or letter == "u":
        vowels += 1
print("vowels count:", vowels)
