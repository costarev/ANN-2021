# output: <tuple> (a, b)
# a - expected value
# b - standard deviation
def m_and_sd(array):
    m = sum(array) / float(len(array))
    sd = (sum([(xi - m) ** 2 for xi in array]) / len(array)) ** 0.5
    return m, sd


input_array = []
print("input: floats separated by space")
try:
    input_array = list(map(float, input().split()))
except ValueError:
    print("incorrect input, exiting...")
    exit()

res = m_and_sd(input_array)
print("(expected value, standard deviation)")
print(res)
print(type(res))
