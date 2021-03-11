import numpy as np
import pandas as pd

def matr(matrix, shape, fill, position):
    new_matrix = np.ones(shape, dtype=matrix.dtype)*fill
    pos  = np.array(list(position)).astype(int)
    new_matrixs = np.array(list(new_matrix.shape)).astype(int)
    matrixs = np.array(list(matrix.shape)).astype(int)

    new_matrix_start = np.zeros((len(shape),)).astype(int)
    new_matrix_stop  = np.array(list(shape)).astype(int)
    matrix_start = (pos - new_matrixs//2)
    matrix_stop  = (pos + new_matrixs//2)+new_matrixs%2

    new_matrix_start = (new_matrix_start - np.minimum(matrix_start, 0)).tolist()
    matrix_start = (np.maximum(matrix_start, 0)).tolist()
    new_matrix_stop = np.maximum(new_matrix_start, (new_matrix_stop - np.maximum(matrix_stop-matrixs,0))).tolist()
    matrix_stop = (np.minimum(matrix_stop,matrixs)).tolist()

    new_matrix_list = [slice(start,stop) for start,stop in zip(new_matrix_start,new_matrix_stop)]
    matrix_list = [slice(start,stop) for start,stop in zip(matrix_start,matrix_stop)]
    new_matrix[new_matrix_list] = matrix[matrix_list]
    output = pd.DataFrame(data=new_matrix.astype(int))
    output.to_csv('output.csv', sep = ';', header = False, index=False)

try:
    fill = int(input('fill = '))            #fill
    pos1 = int(input('pos1 = '))            #центральный элем
    pos2 = int(input('pos2 = '))            #центральный элем
    shape1 = int(input('shape1 = '))        #часть матрицы
    shape2 = int(input('shape2 = '))        #часть матрицы
    shape = (shape1,shape2)                               
    position = (pos1,pos2)     
    matrix = np.genfromtxt('data.csv', dtype='int', delimiter=';')
    matr(matrix, shape, fill, position)
except:
   print('Wrong input')