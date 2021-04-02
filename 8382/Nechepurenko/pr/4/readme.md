# Практическая работа №4

## Задание

Необходимо реализовать нейронную сеть вычисляющую результат заданной логической операции. 
Вариант 6.

`(a and not b) or (c xor b)`

Затем реализовать функции, которые будут симулировать работу построенной модели. 
Функции должны принимать тензор входных данных и список весов.
Должно быть реализовано 2 функции:

1. Функция, в которой все операции реализованы как поэлементные операции над тензорами
2. Функция, в которой все операции реализованы с использованием операций над тензорами из NumPy

Для проверки корректности работы функций необходимо:

1. Инициализировать модель и получить из нее веса
2. Прогнать датасет через не обученную модель и реализованные 2 функции. Сравнить результат.
3. Обучить модель и получить веса после обучения
4. Прогнать датасет через обученную модель и реализованные 2 функции. Сравнить результат.

## Выполнение работы

В дополнение к функциям, описанным в материалах к практической работе, были
добавлены еще несколько, например, функция наивного перемножения двух матриц.

```python3
def naive_matrix_matrix_dot(lhs_matrix: np.array, rhs_matrix: np.array) -> np.array:
    assert len(lhs_matrix.shape) == 2
    assert len(rhs_matrix.shape) == 2
    assert lhs_matrix.shape[1] == rhs_matrix.shape[0]
    result_matrix = np.zeros((lhs_matrix.shape[0], rhs_matrix.shape[1]))
    for i in range(lhs_matrix.shape[0]):
        for j in range(rhs_matrix.shape[1]):
            for k in range(lhs_matrix.shape[1]):
                result_matrix[i, j] += lhs_matrix[i, k] * rhs_matrix[k, j]
    return result_matrix
```

Будем рассматривать следующую модель:
```python3
def build_model() -> Sequential:
    sequential_model = Sequential()
    sequential_model.add(Dense(16, activation='relu', input_shape=(3,)))
    sequential_model.add(Dense(8, activation='relu'))
    sequential_model.add(Dense(16, activation='relu'))
    sequential_model.add(Dense(8, activation='relu'))
    sequential_model.add(Dense(1, activation='sigmoid'))
    return sequential_model
```

Функции симуляции выглядят следующим образом
```python3
def naive_simulation(data: np.array, input_layers: List) -> np.array:
    layers = input_layers.copy()
    last_layer = layers.pop()
    for layer in layers:
        data = make_naive_layer_iteration(data, layer, naive_relu)
    return make_naive_layer_iteration(data, last_layer, naive_sigmoid)
...
def np_simulation(data: np.array, input_layers: List) -> np.array:
    layers = input_layers.copy()
    last_layer = layers.pop()
    for layer in layers:
        data = make_np_layer_iteration(data, layer, lambda x: np.maximum(x, 0))
    return make_np_layer_iteration(data, last_layer, lambda x: 1 / (1 + np.exp(-x)))
```

Выделяем последний слой, итерируясь по оставшимся, применяем шаг слоя алгоритма.
Для наивной реализации он выглядит следующим образом:
```python3
def make_naive_layer_iteration(data: np.array, layer: Dense, activation_fn: Callable) -> np.array:
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    return activation_fn(naive_add_matrix_and_vector(naive_matrix_matrix_dot(data, weights), biases))

```

Затем применяем последний слой и возвращаем ответ.

## Результат работы программы
```
Before fitting
Naive	[0.5]	[0.508]	[0.504]	[0.511]	[0.503]	[0.504]	[0.505]	[0.507]
Numpy	[0.5]	[0.508]	[0.504]	[0.511]	[0.503]	[0.504]	[0.505]	[0.507]
Keras	[0.5]	[0.508]	[0.504]	[0.511]	[0.503]	[0.504]	[0.505]	[0.507]
-----------
Naive rounded	[0.]	[1.]	[1.]	[1.]	[1.]	[1.]	[1.]	[1.]
Numpy rounded	[0.]	[1.]	[1.]	[1.]	[1.]	[1.]	[1.]	[1.]
Keras rounded	[0.]	[1.]	[1.]	[1.]	[1.]	[1.]	[1.]	[1.]
-----------------------------------
After fitting:
Naive	[0.055]	[0.99]	[0.995]	[0.003]	[0.996]	[0.994]	[0.999]	[0.003]
Numpy	[0.055]	[0.99]	[0.995]	[0.003]	[0.996]	[0.994]	[0.999]	[0.003]
Keras	[0.055]	[0.99]	[0.995]	[0.003]	[0.996]	[0.994]	[0.999]	[0.003]
-----------
Naive rounded	[0.]	[1.]	[1.]	[0.]	[1.]	[1.]	[1.]	[0.]
Numpy rounded	[0.]	[1.]	[1.]	[0.]	[1.]	[1.]	[1.]	[0.]
Keras rounded	[0.]	[1.]	[1.]	[0.]	[1.]	[1.]	[1.]	[0.]
-----------------------------------
Correct: 	0	1	1	0	1	1	1	0

```