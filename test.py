n = 64
f = 16

for k in range(1, 11):
    a = k * (3*n + n-f)
    b = k * (2*(n + k*f))
    print((a-b)/n/k, a, b)

