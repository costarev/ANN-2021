import sys
import argparse
import math
from typing import List

def main():
    lhs_vector, rhs_vector = input_vectors(separator=args.sep)
    print(vectors_stdev(lhs_vector, rhs_vector))

def input_vectors(count: int = 2, separator: str = ' ') -> List[List[int]]:
    assert count > 0, 'Count parameter must be positive'
    input_vector = lambda: list(map(int, input().split(sep=separator)))
    vectors = []
    try:
        for _ in range(count):
            vectors.append(input_vector())
    except Exception as e:
        print(f'Exception occured: {e}', file=sys.stderr)
        exit(1)
    else:
        return vectors

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

def run_tests():  # terrible, but idc
    tests = [
        [[[1, 0, 1], [1, 1, 0]], 0.816],
        [[[0, 0, 0], [1, 2, 3]], 2.160],
        [[[-1, 0, 1], [1, 1, 1]], 1.290],
    ]
    confidence_eps = 1e-3
    for vectors, answer in tests:
        output = vectors_stdev(*vectors)
        print(f'Output: {output}, expected: {answer}')
        assert math.isclose(output, answer, rel_tol=confidence_eps)
    

def process_args():
    parser = argparse.ArgumentParser('Pr1 5th task')
    parser.add_argument('--test', help='Specify this for testing', action='store_true')
    parser.add_argument('--sep', help='Specify separator (space by default)', default=' ')
    return parser.parse_args()

def is_testing_mode() -> bool:
    return args.test

if __name__ == '__main__':
    args = process_args()
    if is_testing_mode():
        run_tests()
    else:
        main()