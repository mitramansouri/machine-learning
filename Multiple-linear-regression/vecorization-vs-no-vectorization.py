import numpy as np
import timeit
# Init valuse
x = np.array([1,2,3,4,12,121,5,8,9,9,9,9,8,52,2,25])
w = np.array([5,6,7,8,12,121,5,8,9,9,9,9,8,52,2,25])
n = len(x)
b = 9
# without vectorization
def without_vec():
    f = 0
    for j in range(n):
        f = f + w[j]*x[j]
    f = f + b
    print(f)

execution_time = timeit.timeit(without_vec, number=1)
print(f"Execution time: {execution_time:.6f} seconds")

# with vectorization
def with_vec():
    vf = np.dot(w,x) + b
    print(vf)
execution_time = timeit.timeit(with_vec, number=1)
print(f"Execution time: {execution_time:.6f} seconds")


    
