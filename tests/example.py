from interpreter import *

p = Permutation([1, 3, 2, 4, 5, 6])
q = Permutation([5, 3, 2, 6, 4, 1])

print(p * q)
q = q * q * q
print(inv(p * (q * p)))

v = q(p([100, -2, 0, 8, -5, 1]))
print(v)