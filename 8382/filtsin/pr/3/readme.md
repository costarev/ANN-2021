Задача 2 
(В предположении, что размерность чисел ограничивается 8 байтами)

1. Считываем из файла вектор (n,)

```python
np.fromfile("file.txt", sep=',', dtype=np.uint64)
```

2. С помощью newaxis получаем вектор (n, 1)

```python
np.array(np.fromfile("file.txt", sep=',', dtype=np.uint64)[:, np.newaxis]
```

3. Говорим numpy, что содержимое следует расценивать как 8 незнаковых чисел длиной 8 байт в представлении be

```python
np.array(np.fromfile("file.txt", sep=',', dtype=np.uint64)[:, np.newaxis], dtype='>u8')
```

4. Делаем вьюху на содержимое как на uint8 (в be! см. п. 3)

```python
np.array(np.fromfile("file.txt", sep=',', dtype=np.uint64)[:, np.newaxis], dtype='>u8').view(np.uint8)
```

5. Наконец вызываем unpackbits на полученном объекте