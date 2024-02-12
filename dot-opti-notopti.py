import numpy as np
import time

np.random.seed(1)
a = np.random.rand(1000000000) # very large array
b = np.random.rand(1000000000)

def dot_not_opti(a, b):
    '''
    Compute the dot product of two vectors in a not optimise way
        
        args:
            a (ndarray (n,)): input vector
            b (ndarray (n,)): input vector, w/ same dimension as a
        returns:
            x (scalar):
    '''
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

def dot_opti(a, b):
    '''
    Compute the dot product of two vectors via numpy dot function, so it's opti
    
        args:
            a (ndarray (n,)): input vector
            b (ndarray (n,)): input vector, w/ same dimension as a
        returns:
            np.dot(a, b)
    '''
    return np.dot(a, b)

# opti, result is around 495 ms
tic = time.time()
c = dot_opti(a, b)
toc = time.time()

print(f"opti: {c:.4f}") # c:.4f format c to floating number w/ 4 decimal
print(f"time: {1000*(toc-tic):.4f} ms")

# not opti, result is around 212416 ms
tic = time.time()
c = dot_not_opti(a, b)
toc = time.time()

print(f"not opti: {c:.4f}")
print(f"time: {1000*(toc-tic):.4f} ms")

# remove the large arrays
del(a)
del(b)
