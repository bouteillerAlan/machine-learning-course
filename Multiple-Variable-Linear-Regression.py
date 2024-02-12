import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

''' TRAINING DATA
| size (sqft) | bedrooms | floors | age of house | price (1000s dols) |
|-------------|----------|--------|--------------|--------------------|
|     2104    |     5    |    1   |      45      |        460         |
|     1416    |     3    |    2   |      40      |        232         |
|     852     |     2    |    1   |      35      |        178         |
|-------------|----------|--------|--------------|--------------------|
'''

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

def predict(x, w, b):
    '''
    Single predict using linear regression
    Args:
        x (ndarray): shape (n,) example w/ multiple features
        w (ndarray): shape (n,) model parameters
        b (scalar): model parameter
    Returns:
        p (scalar): prediction
    '''
    
    return np.dot(w, x) + b

# example w/ the first row and a good precalculated w and b
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

x_vec = X_train[0,:]
f_wb = predict(x_vec, w_init, b_init)

# print the result
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}, expected result is {y_train[0]}")
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")


