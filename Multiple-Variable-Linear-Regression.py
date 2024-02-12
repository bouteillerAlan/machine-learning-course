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
        cost = cost + (predict_i - y[i])**2 # give the diff between the predict Ŷ and the example value y[i] and calcul the cost
    
    cost = cost / (2 * m) # implement the 1/2m part of the formula for avoiding big result
    return cost

def compute_gradient(X, y, w, b): # GRADIENT DESCENT DERIVATIVE / kind of cost fc for gradient descent
    # w.r.t. stands for "with respect to," which is a mathematical way
    # of stating that you're examining how a function changes as you 
    # vary a particular parameter or variable
    '''
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar) : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b.
    '''
    
    m,n = X.shape # nb of examples/lines, nb of features
    # dj stand for derivative of the fc J
    dj_dw = np.zeros((n,)) # an array of n 0
    dj_db = 0.
    
    for i in range(m): # line
        err = predict(X[i], w, b) - y[i] # give the diff between the predict Ŷ and the example value y[i]
        
        for j in range(n): # column
            dj_dw[j] = dj_dw[j] + err * X[i, j] # derivative for w the 1/m part is done at the end because we need the sum
        
        dj_db = dj_db + err # derivative for b the 1/m part is done at the end because we need the sum
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_db, dj_dw


    




'''
EXAMPLE FOR EACH FUNCTION BELOW
'''
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
b_bad = 0
w_bad = np.array([ 0, 1, 2, 3])

# MODEL ---- example w/ the first row and a good and a bad precalculated w and b
x_vec = X_train[0,:]
f_wb = predict(x_vec, w_init, b_init)
f_wb_bad = predict(x_vec, w_bad, b_bad)
print("\x1b[1;34m--- prediction example ---\x1b[0m")
print(f"x_vec shape \x1b[1;33m{x_vec.shape}\x1b[0m, x_vec value: \x1b[1;33m{x_vec}\x1b[0m, expected result is \x1b[1;33m{y_train[0]}\x1b[0m")
print(f"f_wb shape \x1b[1;33m{f_wb.shape}\x1b[0m, prediction: \x1b[1;33m{f_wb}\x1b[0m")
print(f"f_wb_bad shape \x1b[1;33m{f_wb_bad.shape}\x1b[0m, prediction: \x1b[1;33m{f_wb_bad}\x1b[0m")

# COST FC ---- example w/ a good and a bad precalculated w and b
cost = compute_cost(X_train, y_train, w_init, b_init)
cost_bad = compute_cost(X_train, y_train, w_bad, b_bad)
print("\x1b[1;34m--- cost fc example ---\x1b[0m")
print(f"cost at optimal \x1b[1;33m{cost}\x1b[0m")
print(f"cost at bad \x1b[1;33m{cost_bad}\x1b[0m")

# GD DERIVATIVE ---- example w/ a good and a bad precalculated w and b
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
tmp_dj_db_bad, tmp_dj_dw_bad = compute_gradient(X_train, y_train, w_bad, b_bad)
print("\x1b[1;34m--- gd deriv. example ---\x1b[0m")
print(f"gd deriv. at optimal \x1b[1;33m{tmp_dj_db, tmp_dj_dw}\x1b[0m")
print(f"gd deriv. at bad \x1b[1;33m{tmp_dj_db_bad, tmp_dj_dw_bad}\x1b[0m")

