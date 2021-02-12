def count_letters(text:str, chars:list=['a','e','i','o','u']) -> map:
    """Count times each character occurs in the text"""
    result_counts = {item: 0 for item in chars}
    for ch in text:
        if ch in result_counts:
            result_counts[ch] += 1
    return result_counts

if __name__ == '__main__':
    str_ = input("Your text: ")
    print(count_letters(str_))