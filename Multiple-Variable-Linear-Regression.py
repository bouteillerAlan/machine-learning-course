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

def predict(x, w, b): # LINEAR REGRESSION - MODEL
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

def compute_cost(X, y, w, b): # COST FUNCTION
    '''
    Compute cost
    Args:
        X (ndarray (m,n)): Data, m examples w/ n features
        y (ndarray (m,)): target value
        w (ndarray (n,)): model params
        b (scalar): model params
    Returns:
        cost (scalar): cost
    '''
    
    m = X.shape[0] # total training example
    cost = 0.0
    
    for i in range(m):
        predict_i = predict(X[i], w, b) # give the prediction Ŷ for the current line i
        cost = cost + (predict_i - y[i])**2 # give the diff between the predict Ŷ and the example value y[i]
    
    cost = cost / (2 * m) # implement the 1/2m part of the formula for avoiding big result
    return cost





'''
EXAMPLE FOR EACH FUNCTION BELOW
'''
# MODEL ---- example w/ the first row and a good precalculated w and b
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

x_vec = X_train[0,:]
f_wb = predict(x_vec, w_init, b_init)
print("\x1b[1;34m--- prediction example start ---\x1b[0m")
print(f"x_vec shape \x1b[1;33m{x_vec.shape}\x1b[0m, x_vec value: \x1b[1;33m{x_vec}\x1b[0m, expected result is \x1b[1;33m{y_train[0]}\x1b[0m")
print(f"f_wb shape \x1b[1;33m{f_wb.shape}\x1b[0m, prediction: \x1b[1;33m{f_wb}\x1b[0m")
print("\x1b[1;34m---  prediction example end  ---\x1b[0m")

cost = compute_cost(X_train, y_train, w_init, b_init)
print("\x1b[1;34m--- cost fc example start ---\x1b[0m")
print(f"cost at optimal w \x1b[1;33m{cost}\x1b[0m")
print("\x1b[1;34m---  cost fc example end  ---\x1b[0m")
