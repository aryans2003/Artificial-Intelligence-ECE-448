import numpy as np
from planar_arm import Arm

# -------------- setup linear regression --------------

# return x,y
# x is a numpy array of shape (num_samples, 1)
#   x values should be uniformly distributed between x_range[0] and x_range[1]
# y is a numpy array of shape (num_samples, 1)
# y = slope * x + intercept + uniform_noise
#   where uniform_noise is uniformly distributed between -noise and noise
def create_linear_data(num_samples, slope, intercept, x_range=[-1.0, 1.0], noise=0.1):
    # x should be uniformly samples between x_range[0] and x_range[1]
    # you should create num_samples x values
    # np.random.rand(num_samples, 1) generates random numbers uniformly distributed between 0 and 1
    # (x_range[1] - x_range[0] calculates width of desired range)
    # the addition to x_range[0] shifts the scaled random numbers to begin at the lower bound
    # the multiplication scales the random numbers within the desired range
    x = x_range[0] + (x_range[1] - x_range[0]) * np.random.rand(num_samples, 1)
    
    # compute y = slope * x + intercept
    y = slope * x + intercept
    
    # add some uniform noise to y sampled between -noise and noise
    uniform_noise = np.random.uniform(-noise, noise, (num_samples, 1))
    y += uniform_noise
    
    # return x, y
    return x, y

# return the modified features for simple linear regression
# x is a numpy array of shape (num_samples, num_features)
# return a numpy array of shape (num_samples, num_features+1)
#   where the last column is all ones
def get_simple_linear_features(x):
    
    # initialize X with a numpy array of shape (num_samples, num_features+1)
    # num_samples is x.shape[0]
    # num_features is x.shape[1]
    # let's also initialize it to ones right now off the bat
    X = np.ones((x.shape[0], x.shape[1] + 1))
    # loop thru rows
    for i in range(x.shape[0]):
        # loop thru first columns
        for j in range(x.shape[1]):
            # set each value in the first columns to x, note we still have second as 1s
            X[i, j] = x[i, j]
    # return modified features
    return X
    
    

# return the prediction for linear regression given x and A
# x is a numpy array of shape (num_samples, num_features)
# A is a numpy array of shape (num_modified_features, 1)
# get_modified_features is a function that takes in x and returns the modified features
#   which have shape (num_samples, num_modified_features)
#   for example get_simple_linear_features
def linear_prediction(x, A, get_modified_features):
    # use get_modified_features function to transform x into feature matrix X
    X = get_modified_features(x)
    
    # perform matrix multiplication between X and A to get predicted values for Y
    Y = np.dot(X, A)
    
    # return Y (predictions)
    return Y

# return the mean squared error loss
# y_pred is a numpy array of shape (num_samples, 1)
# y_true is a numpy array of shape (num_samples, 1)
def mse_loss(y_pred, y_true):
    # MSE = (1/N) sum (of i = 1 to N) (y_i - y_1')^2
    # from stackoverflow, we can do mse = (np.square(matrix A - matrix B)).mean to compute MSE
    return (np.square(y_pred - y_true)).mean()

# return the model error for linear regression
# NOTE: the model error here is just the loss function, 
# in logistic regression later the model error and loss will be different...
def compute_model_error(x, y, A, get_modified_features):
    return mse_loss(linear_prediction(x, A, get_modified_features), y)

# return matrix A of parameters for linear regression, A has shape (num_modified_features, 1)
#   in particular you should compute the analytical solution A for y = A * X
#   i.e., A = (X^T * X)^-1 * X^T * y
# X is a numpy array of shape (num_samples, num_modified_features)
# y is a numpy array of shape (num_samples, 1)
def analytical_linear_regression(X, y):
    # A = (X^T * X)^-1 * X^T * y
    # part 1 is (X^T * X)
    part1 = np.dot(X.T, X)
    # part 2 is (X^T * X)^-1
    part2 = np.linalg.inv(part1)
    # part 3 is (X^T * X)^-1 * X^T
    part3 = np.dot(part2, X.T)
    # finally, return (X^T * X)^-1 * X^T * y
    return np.dot(part3, y)

# -------------- gradient descent for linear regression --------------

# return the gradient of the MSE loss function for linear regression
#   MSE loss is: 1/N * ||Y - XA||_2^2, i.e., 1/N * (Y - XA)^T * (Y - XA)
#   and the gradient is: 2 * X^T * (X * A - Y) / N
#   where N is the number of samples
# A is a numpy array of shape (num_modified_features, 1)
# X is a numpy array of shape (num_samples, num_modified_features)
# y is a numpy array of shape (num_samples, 1)
def get_linear_regression_gradient(A, X, y):
    # N is number of samples
    N = len(y)
    # (X * A - y)
    part1 = np.dot(X, A) - y
    # simplify equation to (2/N) * X^T * part1
    return (2 / N) * np.dot(X.T, part1)

# return matrix A of parameters, A has shape (num_modified_features, 1)
#   in particular run gradient descent with learning rate learning_rate for num_iterations
# A_init is a numpy array of shape (num_modified_features, 1)
# get_gradient is a function that returns the gradient of the loss function with respect to A
#   i.e., get_gradient = lambda A: get_linear_regression_gradient(A, X, y) 
def gradient_descent(get_gradient, A_init, learning_rate, num_iterations):
    # A and A_init has same shape so just make a copy
    A = A_init.copy()
    # run gradient descent with learning rate for num_iterations
    for iterations in range(num_iterations):
        # compute gradient use get_gradient
        gradient = get_gradient(A)
        # update A using gradient descent update rule
        # equation is: A = A - learning_rate * gradient
        A = A - learning_rate * gradient
    return A

# -------------- stochastic gradient descent for linear regression --------------

# return matrix A of parameters, A has shape (num_modified_features, 1)
#   in particular run stochastic gradient descent with learning rate learning_rate 
#   for num_epochs epochs (one epoch is one pass through the entire dataset) with batch size batch_size
#   HINT: make sure to shuffle the indices of the dataset before EACH epoch
#       - you may find np.random.permutation useful
# A_init is a numpy array of shape (num_modified_features, 1)
# get_batch_gradient is a function that returns the gradient of the loss function with respect to A
#   for a specific batch of indices, 
#   i.e., get_batch_gradient = lambda A, indices: get_linear_regression_gradient(A, X[indices], y[indices])
# data_size is the number of samples in the dataset
# batch_size is an integer representing the number of samples to use in each iteration
#   1 <= batch_size <= data_size
def stochastic_gradient_descent(get_batch_gradient, A_init, learning_rate, num_epochs, data_size, batch_size):
    # A has same as A_init
    A = A_init.copy()
    # for epochs
    for epochs in range(num_epochs):
        # shuffle indices before each epoch using datasize and given np function
        indices = np.random.permutation(data_size)
        # for each batch
        for batch in range(0, data_size, batch_size):
            # compute batch indices
            # first we check the ending index for the batch
            ending = batch + batch_size
            # if it is > data_size...
            if ending > data_size:
                # update it to data size
                ending = data_size
            # use the ending index to compute batch indices by slicing the indices array
            batch_indices = indices[batch:ending]
            # compute stochastic gradient descent
            gradient = get_batch_gradient(A, batch_indices)
            # update A using gradient descent update rule
            # equation is: A = A - learning_rate * gradient
            A = A - learning_rate * gradient
    # return A
    return A

# -------------- polynomial regression for sine function --------------

# return x, y for the sine function with noise
# x is a numpy array of shape (num_samples, 1)
#   x values should be uniformly distributed between x_range[0] and x_range[1]
# y is a numpy array of shape (num_samples, 1)
# y = sin(x) + uniform_noise
# uniform_noise is uniformly distributed between -noise and noise
def create_sine_data(num_samples, x_range=[0.0, 2*np.pi], noise=0.1):
    #   x values should be uniformly distributed between x_range[0] and x_range[1]
    # using same code from first function....
    x = x_range[0] + (x_range[1] - x_range[0]) * np.random.rand(num_samples, 1)
    # uniform_noise is uniformly distributed between -noise and noise
    # also using same code from first function...
    uniform_noise = np.random.uniform(-noise, noise, (num_samples, 1))
    # y = sin(x) + uniform_noise
    y =  np.sin(x) + uniform_noise
    # return x, y
    return x, y

# return the modified polynomial features for doing linear regression
#   i.e., polynomial regression: y = a_n * x^n + ... + a_1 * x + a_0
# x is a numpy array of shape (num_samples, num_features)
#   - NOTE: num_features is 1 for this problem but later we will use more features
# return a numpy array of shape (num_samples, num_features * (degree + 1))
#   i.e., return X = [x^n, x^(n-1), ..., x, 1]
def get_polynomial_features(x, degree):
    # initialize empty list to store polynomial features for all samples
    features = [] 
    # for each sample xi... (samples, features in xi so loop thru samples first)
    for xi in x:
        # init empty list to store powers of xi
        xi_features = []
        # now loop thru features
        for xi_val in xi:
            # compute [x^n, x^(n-1), ..., x, 1]
            # loop thru [degree, 1] --> descending order
            for i in range(degree, 0, -1):
                # compute current degree as we loop down and store in xi_features
                xi_features.append(xi_val**i)
            # after xi_features populated, add the 1 for x^0 as any #^0 = 1
            xi_features.append(1)
        # append finalized xi_features to our original features empty list
        features.append(xi_features)
    # return features but convert to np array
    return np.array(features)

# -------------- inverse kinematics via gradient descent --------------

# return the loss for the inverse kinematics problem, 
#   i.e., the (2 dimensional) euclidean distance between the end effector and the goal
#   you can get the end effector position by calling arm.forward_kinematics(config)[-1]
# arm is an Arm object
# config is a numpy array of shape (num_joints,)
# goal is a numpy array of shape (2,)
def ik_loss(arm : Arm, config, goal):
    # get the end effector position
    end_effector_pos = arm.forward_kinematics(config)[-1]
    # compute 2 dimensional euclidean distance 
    # square root of (y2-y1)^2 + (x2-x1)^2
    distance = (end_effector_pos[1] - goal[1])**2 + (end_effector_pos[0] - goal[0])**2
    return np.sqrt(distance)

# we provide a more complex loss function that includes obstacles
# this loss is high when the arm is close to an obstacle
# obstacles is a list of obstacles, each obstacle is a numpy array of shape (num_obstacles, 3) 
# where each obstacle is a circle with (x,y,radius)
def ik_loss_with_obstacles(arm : Arm, config, goal, obstacles):
    # first compute the ik loss without obstacles
    ee_loss = ik_loss(arm, config, goal)
    # now compute the obstacle loss as a sum of harmonic losses (1/distance)
    workspace_config = arm.forward_kinematics(config)
    total_obstacle_loss = 0
    for obstacle in obstacles:
        # find the closest joint to the obstacle 
        # (technically we should do line-segment to circle distance)
        obstacle_dist = np.min(np.linalg.norm(workspace_config - obstacle[:2], axis=1))
        # if the joint is inside the obstacle, return infinity
        if obstacle_dist < obstacle[2]:
            return np.inf
        # otherwise, compute the harmonic loss
        total_obstacle_loss += 1 / (obstacle_dist - obstacle[2])
        # we could instead use a quadratic penalty...
        # total_obstacle_loss += -(obstacle_dist - obstacle[2])**2
    return ee_loss + total_obstacle_loss

# given a configuration, sample nearby points and return them
#   return a numpy array of shape (num_samples, num_joints)
# num_samples is the number of samples to return
# config is a numpy array of shape (num_joints,)
# epsilon is the max distance to sample nearby points
#   points should be sampled uniformly a distance epsilon from config (in each dimension)
# HINT: array broadcasting is your friend, and if you don't know what this means look it up
def sample_near(num_samples, config, epsilon=0.1):
    # array broadcasting --> the smaller array is broadcast across the larger array so that they have compatible shapes
    # this means there is no need to create X beforehnd
    # get num_joints
    num_joints = config.shape[0]
    # points should be sampled uniformly a distance epsilon from config in each dim
    # using similar formula from create_sine_data...
    uniform = np.random.uniform(-epsilon, epsilon, (num_samples, num_joints))
    # append to X
    X = config + uniform
    # return X
    return X

# estimate the gradient of the loss function at config by:
#   1. sampling nearby points 
#   2. picking the direction of MAXIMUM loss
#   3. normalize this vector to return a UNIT vector of shape (num_features,)
# loss is a function that takes in a configuration and returns a scalar loss
# config is a numpy array of shape (num_features,)
# num_samples is the number of samples to use to estimate the gradient (use sample_near)
def estimate_ik_gradient(loss, config, num_samples):
    # 1. sample nearby ports (use sample_near(num_samples, config, epsilon))
    # we dont have epsilon defined here so lets just do that as same one in function header
    epsilon = 0.1
    sampling = sample_near(num_samples, config, epsilon)
    # compute loss for each sample
    # create losses array using num_samples as size
    losses = np.empty(num_samples)
    # loop thru samples
    for i, sample in enumerate(sampling):
        # append curr sample loss to our losses array
        losses[i] = loss(sample)
    
    #2. pick dir of max loss
    # the index of the maximum loss is the max of losses array
    loss_index = np.argmax(losses)
    # the config is the sample at that index
    loss_config = sampling[loss_index]
    # compute dir
    dir = loss_config - config
    
    # 3. normalize vector to get gradient
    # normalize using direction / normalize(dir)
    gradient = dir / np.linalg.norm(dir)
        
    return gradient

# -------------- logistic regression for provided data --------------

# compute the average error rate given the predictions and true labels
#   a prediction is 1 if y_pred > 0.5 and 0 otherwise
# y_pred is a numpy array of probabilities of shape (num_samples, 1)
# y_true is a numpy array of 0's and 1's of shape (num_samples, 1)
def logistic_error(y_pred, y_true):
    # init list to store predictions after rounded to 0 or 1
    predictions = []
    # loop thru predictions
    for prediction in y_pred:
        # if > 0.5...
        if prediction > 0.5:
            # append 1
            predictions.append(1)
        # else if less...
        else:
            # append 0
            predictions.append(0)
    # convert predictions lsit to numpy array for comparison
    predictions = np.array(predictions)
    # calculate error rate by comparing predictions to true labels
    # init var for error count
    error_count = 0
    # get number of samples
    num_samples = len(y_true)
    # loop thru samples
    for i in range(num_samples):
        # check if prediction matches true label
        if predictions[i] != y_true[i]:
            # increase error_count
            error_count += 1
    # compute error rate from errors / num_samples
    # which corresponds to y_pred / y_true
    error_rate = error_count / num_samples
    return error_rate

# logistic regression prediction is the sigmoid of the linear prediction
#   i.e., y_pred = 1 / (1 + exp(-X * A))
#   you could (should) use linear_prediction to get the linear prediction, i.e., X * A
# x is a numpy array of shape (num_samples, num_features)
# A is a numpy array of shape (num_modified_features, 1)
# get_modified_features is a function that takes in x and returns the modified features
    # you could should use linear_prediction to get the linear prediction, i.e., X * A
    # def linear_prediction x, A, get_modified_features
    # linear_pred = linear_prediction(x, A, get_modified_features)
    # y_pred = 1 / (1 + exp(-linear_pred))
    # return 1 / (1+ np.exp(-linear_pred))
def logistic_prediction(x, A, get_modified_features):
    # get X from modified features of x
    X = get_modified_features(x)
    # calculate linear_pred as = X * a
    # NOTE: I TRIED USING linear_prediction() function but it broke test 1 case??
    linear_pred = np.dot(X, A)
    # compute y_pred = 1 / (1 + exp(-X * A))
    # except now (-X * A) is just -linear_pred
    return 1 / (1 + np.exp(-linear_pred))

# the logistic loss function for binary classification with y_true in {0,1}
#   loss = - sum_i (y_true[i] * log(y_pred[i]) + (1 - y_true[i]) * log(1 - y_pred[i]))
# y_pred is a numpy array of probabilities of shape (num_samples, 1)
# y_true is a numpy array of 0's and 1's of shape (num_samples, 1)
# NOTES:
#   - don't use a for loop!
#   - log(0) is undefined, so you should clip y_pred to be between epsilon and 1-epsilon for small epsilon
def logistic_loss(y_pred, y_true):
    # let's use epsilon from prev function
    epsilon = 1e-15
    # clip y_pred between epsilon and 1 - small epsilon for a small epsilon
    y_pred_clip = np.clip(y_pred, epsilon, 1 - epsilon)
    # compute loss function 
    # loss = - sum_i (y_true[i] * log(y_pred[i]) + (1 - y_true[i]) * log(1 - y_pred[i]))
    # instead of y_pred[i] use the clipped value
    # no for loop so no indices
    return -np.sum(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))

# return the gradient of the logistic loss function for logistic regression
#   the gradient is: X^T * (y_pred - y_true)
#   you should use logistic_prediction to get y_pred with modified_features as the identity, 
#   i.e., y_pred = logistic_prediction(X, A, modified_features = lambda x: x)
# A is a numpy array of shape (num_modified_features, 1)
# X is a numpy array of shape (num_samples, num_modified_features)
# y is a numpy array of 0's and 1's of shape (num_samples, 1)
def get_logistic_regression_gradient(A, X, y):
    # we want to extract modified_features as modified_features = lambda x: x
    modified_features = lambda x: x
    # use logistic_prediction to get y_pred with modified_features as the identity
    y_pred = logistic_prediction(X, A, modified_features)
    # the gradient is: X^T * (y_pred - y_true)
    gradient = np.dot(X.T, (y_pred - y))
    # return the gradient
    return gradient

# return the modified features for logistic regression
# x is a numpy array of shape (num_samples, num_features)
# return a numpy array of shape (num_samples, num_modified_features)
def get_logistic_regression_features(x):
    # for more complex relationships, we classify using non-linear regression techniques (experimented)
    
    # store all features in list
    features = []
    
    # put x in there to begin
    features.append(x)
    
    # add degree 2
    # degree 1 = x so alr in there
    # degree 2...
    features.append(x**2)
        
    # append trig transformations
    features.append(np.sin(x))
    features.append(np.cos(x))
    
    # add exponential transformation
    features.append(np.exp(x))
    
    # return transformed feature matrix
    return np.column_stack(features)
