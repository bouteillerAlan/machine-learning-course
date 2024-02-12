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

def compute_gradient(X, y, w, b): # GRADIENT DESCENT DERIVATIVE / do the gradient descent for one "path"
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

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): # GRADIENT DESCENT
    '''
    Performs batch gradient descent to learn w and b.
    Updates w and b by taking num_iters gradient steps with learning rate alpha
    !!! HAVE SOME CODE FOR GRAPH AND DEBUG !!!
    
    Args:
      X (ndarray (m,n)) : Data, m examples with n features
      y (ndarray (m,)) : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar) : initial model parameter
      cost_function : function to compute cost
      gradient_function : function to compute the gradient
      alpha (float) : Learning rate
      num_iters (int) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar) : Updated value of parameter
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
    '''
    
    J_history = [] # an array to store cost J and w's at each iteration primarily for graphing later
    p_history = []
    w = copy.deepcopy(w_in) # avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)

        # update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw # first part of the GD algo + derivative
        b = b - alpha * dj_db
      
        # save cost J at each iteration for the graph
        if i < 100000: # prevent resource exhaustion if user put a too big num_iters 
            J_history.append(cost_function(X, y, w, b))
            p_history.append([w,b])

        # print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")
        
    return w, b, J_history, p_history # return final w,b and J/p history for the graph

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

# GRADIENT DESCENT ---- IRL example
print("\x1b[1;34m--- gd irl ---\x1b[0m")
# initialize parameters
initial_w = np.zeros_like(w_init) # create an array of 0 at the same lgt of w_init
initial_b = 0.
# gradient descent settings
iterations = 5000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist, p_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print(f"b,w found by gradient descent: \x1b[1;33m{b_final:0.2f} {w_final}\x1b[0m")
print("\x1b[1;34m--- prediction irl w/ the previous b,w ---\x1b[0m")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: \x1b[1;33m{predict(X_train[i], w_final, b_final):0.2f}\x1b[0m, target value: \x1b[1;33m{y_train[i]}\x1b[0m")

# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()
