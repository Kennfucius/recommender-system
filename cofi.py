########## ---- Collaborative Filtering Model ---- ##########
########## ---- Created by Kenneth Liao (12/1/2017) ---- ##########

# This general recommender system was created using a collaborative filtering
# model. The model utilizes a data matrix X containing users on the 0 axis and
# products (movies, food, merchandise, etc.) on the 1 axis. The values in
# the matrix are the ratings given to each product by users.

# This model is designed to minimize the cost of predictions using the
# fmin_cg function from the scipy.optimize package, which requires computing
# the cost function and gradient at every iteration.



##### Code begins here #####

# While the original matrices may be in a DataFrame format, the model employs
# numpy functions exclusively.
import numpy as np

# get_args takes in exactly 3 arguments: data matrix X, the number of desired
# features, N_features and the desired regularization parameter, Lambda.

def get_args(X, N_features, Lambda):

    X = np.array(X)
    mu = np.nanmean(X, axis=0)
    Y = X - mu # (N_users x N_games)

    N_users = X.shape[0]
    N_games = X.shape[1]

    # Initialize two random matrices to make our initial predictions
    X_init = np.random.rand(N_games, N_features) - 0.5
    Theta_init = np.random.rand(N_users, N_features) - 0.5

    args = (X_init, Y, Theta_init, Lambda, N_users, N_games, N_features, mu)

    return args

# unroll_params takes two matrices (X and Theta), and unrolls them into a
# single end-to-end vector (params).

def unroll_params(X, Theta, order='C'):

    X = np.array(X)
    Theta = np.array(Theta)

    parameters = np.concatenate((X.flatten(order=order),
                                 Theta.flatten(order=order)), axis=0)

    return parameters

# roll_params takes an unrolled vector, params, and reshapes it into the
# matrices X and Theta.

def roll_params(parameters, *args):

    N_users = args[4]
    N_games = args[5]
    N_features = args[6]

    dim1 = N_games*N_features

    X = np.reshape(parameters[0:dim1], (N_games, N_features))
    Theta = np.reshape(parameters[dim1:], (N_users, N_features))

    return X, Theta

# cost_f takes in the parameters vector and computes the model's predictions.
# It then compares the model's predictions to the actual ratings and computes
# a cost associated with the model's current parameters.

def cost_f(parameters, *args):

    Y = args[1]
    Lambda = args[3]

    X, Theta = roll_params(parameters, *args)

    hyp = np.dot(Theta,X.T)
    error = hyp - Y
    error_factor = error.copy() # dimensions (N_games x N_users)
    error_factor[np.isnan(error)] = 0 # Sets all missing values to 0s

    # Compute the COST FUNCTION with REGULARIZATION
    Theta_reg = (Lambda/2) * np.nansum(Theta*Theta)
    X_reg = (Lambda/2) * np.nansum(X*X)

    J = (1/2) * np.nansum(error_factor*error_factor) + Theta_reg + X_reg

    return J

# grad_f calculates the gradients of the cost function w.r.t. X and Theta

def grad_f(parameters, *args):

    Y = args[1]
    Lambda = args[3]

    X, Theta = roll_params(parameters, *args)

    hyp = np.dot(Theta,X.T)
    error = hyp - Y
    error_factor = error.copy() # dimensions (N_games x N_users)
    error_factor[np.isnan(error)] = 0 # Sets all missing values to 0s

    X_grad = np.dot(error_factor.T, Theta) + Lambda*X
    Theta_grad = np.dot(error_factor, X) + Lambda*Theta

    grad = unroll_params(X_grad, Theta_grad)

    return grad

##### Code ends here #####
