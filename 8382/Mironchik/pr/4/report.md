# Мирончик Павел, 8382, вар.5

**(a xor b) and (b xor c)**

Считываем данные

```
data = np.genfromtxt('input_data.csv', delimiter=';')
train_data = data[:, :3]
train_labels = data[:, 3]
```

и прогоняем их через слои тремя способами.

1.  Через numpy

    ```
    def np_run(layers, input):
        res = np.maximum(np.dot(input, layers[0].get_weights()[0]) + layers[0].get_weights()[1], 0)
        res = np.maximum(np.dot(res, layers[1].get_weights()[0]) + layers[1].get_weights()[1], 0)
        res = sigmoid(np.dot(res, layers[2].get_weights()[0]) + layers[2].get_weights()[1])
        return res
    ```

2.  Через кастомную реализацию

    ```
    def naive_run(layers, input):
        res = naive_relu(naive_matrix_matrix_dot(input, layers[0].get_weights()[0]) + layers[0].get_weights()[1])
        res = naive_relu(naive_matrix_matrix_dot(res, layers[1].get_weights()[0]) + layers[1].get_weights()[1])
        res = sigmoid(naive_matrix_matrix_dot(res, layers[2].get_weights()[0]) + layers[2].get_weights()[1])
        return res
    ```

    где `naive_relu` == `np.maximum`, `naive_matrix_matrix_dot` == `np.dot` (равны по выполняемому действию, а не как ссылки, естественно).
    
3. Через Keras

    ```
   model.predict(train_data)
   ```
   
Во всех трех случаях получаем одинаковые значения как до обучения, так и после. 

PS: в принципе 600 эпох слегка многовато, т.к. уже на ~238 эпохе метрика accuracy составляет 1.0. Но зато разница между предсказанным значением и реальным составляет не более 3-4 сотых :)