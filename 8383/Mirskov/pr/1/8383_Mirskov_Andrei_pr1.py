def count_1(n):
	s = 0
	while n:
		s += n & 1
		n >>= 1
	return s

n = int(input('Ведите число '))
print('Битов равных 1:', count_1(n))