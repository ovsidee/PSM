import math as m

def compute_sin(x, n = 50):
    result = 0
    for i in range(n):
        result += (((-1)**i) / m.factorial(2*i + 1)) * (x**(2*i + 1))
    return result

for i in range(0, 20):
    print(f"x = {i} \nour function: {compute_sin(i)} \nbase python function: {m.sin(i)} \n")