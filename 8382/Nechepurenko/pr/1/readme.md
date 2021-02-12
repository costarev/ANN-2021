# Практическая работа № 1

## Задание:
Вариант 5
Написать функцию, которая принимает два целочисленных вектора одинаковой длины и возвращает среднеквадратическое отклонение двух векторов.

## Реализация:
Не совсем понятно, что такое ско двух векторов.
Пусть под этим понимаем корень усредненной суммы квадратов разности компонент.

Считаем два вектора с помощью функции `input_vectors`. В функции `vectors_stdev` подсчитаем
нужную метрику.

```python3
def vectors_stdev(lhs: List[int], rhs: List[int]) -> float:
    assert len(lhs) == len(rhs), "Sizes should be equal"
    assert len(lhs) > 0, "Sizes should be positive"
    try:
        sum_squared_diff = sum(map(lambda x, y: (x - y)**2, lhs, rhs), 0)
        answer = math.sqrt(sum_squared_diff / len(lhs))
    except Exception as e:
        print(f'Exception occured: {e}', file=sys.stderr)
    else:
        return answer
```

## Использование:
Пример:
```
> py main.py
1 1 1
2 2 2
1.0
```

Есть возможность указать разделитель с помощью парамера `sep`
Пример:
```
❯ py main.py --sep=";"
1;2;3
4;5;6
3.0
```

Для запуска тестов необходимо указать флаг `test`
Пример:
```
❯ py main.py --test
Output: 0.816496580927726, expected: 0.816
Output: 2.160246899469287, expected: 2.16
Output: 1.2909944487358056, expected: 1.29
```

## Комментарии:
Тесты убогие, посчитал излишним оборачивать их во что-то нормальное. 
