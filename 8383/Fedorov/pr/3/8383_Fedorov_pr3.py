import numpy as np

#print(np.vstack({(row) for row in data}))

def find_unique_str_np(file_read='test1.csv', file_save='result.csv',
                       dtype_='int', delim=';', console_log=False):
    sep_ = '_________________'

    def save_res(res):
        fmt_ = ''
        if dtype_ == 'int':
            fmt_ = '%i'
        else:
            fmt_ = '%f'
        np.savetxt(file_save, res, delimiter=delim, fmt = fmt_) 
    
    def log_info(text, sep_='', description=''):
        print(description)
        print(text)
        print(sep_)
    
    data = np.genfromtxt(file_read, dtype=dtype_, delimiter=delim)

    if data.size == 0:
        raise ValueError("Empty file!")

    if (len(data.shape) == 1):
        save_res(np.asmatrix(data))
        return data

    res, index = np.unique(data, axis=0, return_index=True)

    if console_log:
        log_info(data, sep_)
        log_info(res, sep_, 'Unique str:')

    save_res(res)
    return res


def naive_find_unique_str(file_read='test1.csv', file_save='result.csv',
                       dtype_='int', delim=';', console_log=False):

    def save_res(res):
        fmt_ = ''
        if dtype_ == 'int':
            fmt_ = '%i'
        else:
            fmt_ = '%f'
        np.savetxt(file_save, res, delimiter=delim, fmt = fmt_) 

    data = np.genfromtxt(file_read, dtype=dtype_, delimiter=delim)

    if data.size == 0:
        raise ValueError("Empty file!")

    if (len(data.shape) == 1):
        save_res(np.asmatrix(data))
        return data
        
    res = []
    for i in range(len(data)-1):
        is_unique = True
        for j in range(i+1,len(data)):
            if np.array_equal(data[i],data[j ]):
                is_unique = False
                break

        if is_unique:
            res.append(data[i])

    res.append(data[len(data)-1])
    res = np.array(res)

    save_res(res)
    return res

if __name__ == '__main__':
    file_task = input("File to read: ")
    file_result = input("File to save: ")
    find_unique_str_np(file_read=file_task, file_save=file_result, console_log=True) 
    naive_find_unique_str(file_read=file_task, file_save=file_result)

   
