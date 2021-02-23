#!/usr/bin/env python
# coding: utf-8

# ### Задание 
# #### Написать функцию, которая находит самое часто встречающееся число в каждой строке матрицы и возвращает массив этих значений

# ### Импорт

# In[1]:


from statistics import mode
import numpy as np


# ### Считаем моду в каждой строке

# In[2]:


def f(x):
    modes = []
    for line in x:
        modes.append(mode(line))
    return modes


# ### Пример выполнения

# In[3]:


m = np.random.randint(-4,4,size=(10,10))
print(m)
f(m)


# In[ ]:




