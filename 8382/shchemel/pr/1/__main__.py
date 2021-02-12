def is_simple(value: str) -> bool:
    return len(set(value)) == len(value)


if __name__ == '__main__':
    input_str = input()
    print(is_simple(input_str))
