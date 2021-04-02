from typing import Tuple

import numpy as np


def load_data(filename: str) -> Tuple[int, int, int, int]:
    return np.fromfile(filename, dtype=int, sep=" ")


def make_chess(m: int, n: int, a: int, b: int) -> np.array:
    ret_array = np.zeros((m, n))
    ret_array[::2, ::2] = a
    ret_array[1::2, 1::2] = a
    ret_array[::2, 1::2] = b
    ret_array[1::2, ::2] = b

    return ret_array


def save_data(filename: str, data: np.array) -> None:
    np.savetxt(filename, data, delimiter=" ", fmt="%d")


if __name__ == '__main__':
    chess = make_chess(*load_data("input.txt"))
    save_data("output.txt", chess)
