### Задание 
#### Написать функцию, которая находит самое часто встречающееся число в каждой строке матрицы и возвращает массив этих значений

### Импорт


```python
from statistics import mode
import numpy as np
```

### Считаем моду в каждой строке


```python
def f(x):
    modes = []
    for line in x:
        modes.append(mode(line))
    return modes
```

### Пример выполнения


```python
#m = np.random.randint(-4,4,size=(10,10))
m = np.loadtxt("v9.txt")
print(m)
np.savetxt("v9_result.txt",f(m))
```

    [[1. 4. 8. 8.]
     [2. 2. 8. 8.]
     [1. 3. 5. 2.]
     [1. 1. 1. 1.]
     [1. 1. 1. 3.]]
    


```python

```
