task1 = lambda s: sum([s.lower().count(c) for c in ('a', 'e', 'i', 'o', 'u')]) if type(s) is str else None

task2 = lambda s: len(s) == len(set(s)) if type(s) is str else None

task3 = lambda x: bin(x)[2:].count('1') if x >= 0 else None

task6 = lambda numbers, is_cp: (sum(numbers) / len(numbers),
                                (sum([(x - sum(numbers) / len(numbers)) ** 2 for x in numbers]) / (
                                        len(numbers) - (not is_cp)))**(0.5))


def task9(n: int):
    if type(n) is not int:
        return "Not int"
    if n < 1:
        return "Must be > 0"
    k = 1
    res = k
    while res < n:
        k += 1
        res += k ** 2
    return k if n == res else "It is impossible"


if __name__ == '__main__':
    assert task1("asdf") == 1
    assert task1("Asdf") == 1
    assert task1("aadf") == 2
    assert task1("sdf") == 0
    assert task1(1) is None

    assert task2("asdf")
    assert not task2("aa")
    assert task2("")
    assert task2(3) is None

    assert task3(-100) is None
    assert task3(10) == 2
    assert task3(2) == 1
    assert task3(7) == 3

    res6 = task6([1, 1, 1, 1], True)
    assert res6[0] == 1
    assert res6[1] == 0
    res6 = task6([1, 2, 3, 2], True)
    assert res6[0] == 2
    assert .707 < res6[1] < .708
    res6 = task6([1, 2, 3, 2], False)
    assert res6[0] == 2
    assert .816 < res6[1] < .817

    assert task9(1) == 1
    assert task9(5) == 2
    assert task9(2) == "It is impossible"
    assert task9(14) == 3
    assert task9(30) == 4
    assert task9("asfd") == "Not int"
    assert task9(-1) == "Must be > 0"
    print("All OK!!!")
