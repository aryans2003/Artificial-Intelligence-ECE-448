import numpy as np
import time

from mp9 import *

def test_create_linear_data(num_samples = 100, slope = 2, intercept = 1, x_range = [-10,10], noise = 0):
    x, y = create_linear_data(num_samples=num_samples, slope=slope, 
                              intercept=intercept, x_range=x_range, noise=noise)
    assert type(x) == np.ndarray and type(y) == np.ndarray
    assert x.shape == (num_samples,1)
    assert y.shape == (num_samples,1)
    assert np.all(x >= x_range[0]) and np.all(x <= x_range[1])
    for i in range(num_samples):
        assert y[i] <= slope*x[i] + intercept + noise and y[i] >= slope*x[i] + intercept - noise

    # check that the samples are random
    x_new, _ = create_linear_data(num_samples=num_samples, slope=slope, 
                                intercept=intercept, x_range=x_range, noise=noise)
    assert not np.all(np.isclose(x, x_new))

    # check that the y values are noisy... this should hopefully catch the common cases
    if noise != 0:
        assert not np.all(np.isclose(y, slope*x + intercept))
        assert not np.all(np.isclose(y, slope*x + intercept + noise))
        assert not np.all(np.isclose(y, slope*x + intercept - noise))
        # NOTE: this may sometimes fail with 1/2^num_samples chance
        assert not np.all(y > slope*x + intercept)
        assert not np.all(y < slope*x + intercept)

    # check that creating 10,000,000 samples is less than a second
    start = time.time()
    x, y = create_linear_data(num_samples=10000000, slope=slope, 
                              intercept=intercept, x_range=x_range, noise=noise)
    assert time.time() - start < 1

def test_get_simple_linear_features(num_samples=10, num_features=1):
    x = np.zeros((num_samples, num_features))
    features = get_simple_linear_features(x)
    assert type(features) == np.ndarray
    assert features.shape == (num_samples, num_features+1)
    assert np.all(features[:,-1] == 1)

def test_linear_prediction():
    x = np.array([[1],[2],[3]])
    A = np.array([[4],[-1]])
    y = linear_prediction(x, A, get_modified_features=get_simple_linear_features)
    assert type(y) == np.ndarray
    assert y.shape == (3,1)
    assert np.all(y == np.array([[3],[7],[11]]))

def test_mse_loss():
    y_true = np.array([[1],[2],[3]])
    y_pred = np.array([[2],[3],[4]])
    loss = mse_loss(y_true, y_pred)
    assert loss == 1

    y_true = np.array([[1],[2],[3]])
    y_pred = np.array([[1],[2],[3]])
    loss = mse_loss(y_true, y_pred)
    assert loss == 0

    y_true = np.array([[1],[2],[3]])
    y_pred = np.array([[3],[2],[1]])
    loss = mse_loss(y_true, y_pred)
    assert np.isclose(loss, 8/3)

def test_analytical_linear_regression():
    x = np.array([[1],[2],[3]])
    y = np.array([[3],[7],[11]])
    X = get_simple_linear_features(x)
    A = analytical_linear_regression(X, y)
    assert type(A) == np.ndarray
    assert A.shape == (2,1)
    assert np.all(np.isclose(A, np.array([[4],[-1]])))

    x, y = create_linear_data(num_samples=100, slope=2, intercept=1, x_range=[-10,10], noise=0)
    X = get_simple_linear_features(x)
    A = analytical_linear_regression(X, y)
    assert np.all(np.isclose(A, np.array([[2],[1]])))

    x, y = create_linear_data(num_samples=100, slope=-3, intercept=2, x_range=[0,5], noise=0.1)
    X = get_simple_linear_features(x)
    A = analytical_linear_regression(X, y)
    # 0.2 is arbitrary... just to make sure it's close
    assert np.all(np.abs(A - np.array([[-3],[2]]) < 0.2))

def test_linear_regression_gradient():
    # basically just test that the gradient at the true A is close to 0
    x, y = create_linear_data(num_samples=100, slope=2, intercept=1, x_range=[-10,10], noise=0)
    X = get_simple_linear_features(x)
    grad = get_linear_regression_gradient(np.array([[2],[1]]), X, y)
    assert type(grad) == np.ndarray
    assert grad.shape == (2,1) # same shape as A
    assert np.linalg.norm(grad) < 0.001

    x, y = create_linear_data(num_samples=100, slope=-10, intercept=5, x_range=[0,5], noise=0.1)
    X = get_simple_linear_features(x)
    grad = get_linear_regression_gradient(np.array([[-10],[5]]), X, y)
    assert np.linalg.norm(grad) < 0.2 # 0.2 is arbitrary... just to make sure it's close enough

def test_gradient_descent():
    x, y = create_linear_data(num_samples=100, slope=2, intercept=1, x_range=[-10,10], noise=0)
    X = get_simple_linear_features(x)
    A_init = np.array([[0],[0]])
    get_gradient = lambda A: get_linear_regression_gradient(A, X, y)
    A = gradient_descent(get_gradient, A_init, learning_rate=0.01, num_iterations=1000)
    assert np.all(np.isclose(A, np.array([[2],[1]]), atol=0.1))

    x, y = create_linear_data(num_samples=100, slope=-10, intercept=5, x_range=[0,5], noise=0.5)
    X = get_simple_linear_features(x)
    A_init = np.array([[0],[0]])
    get_gradient = lambda A: get_linear_regression_gradient(A, X, y)
    A = gradient_descent(get_gradient, A_init, learning_rate=0.01, num_iterations=1000)
    assert np.all(np.isclose(A, np.array([[-10],[5]]), atol=0.5))

def test_stochastic_gradient_descent():
    # run sgd twice - results should be different
    x, y = create_linear_data(num_samples=100, slope=2, intercept=1, x_range=[-10,10], noise=0.1)
    X = get_simple_linear_features(x)
    A_init = np.array([[0],[0]])
    get_batch_gradient = lambda A, indices: get_linear_regression_gradient(A, X[indices], y[indices])
    A1 = stochastic_gradient_descent(get_batch_gradient, A_init, learning_rate=0.01, num_epochs=1, data_size=100, batch_size=10)
    A2 = stochastic_gradient_descent(get_batch_gradient, A_init, learning_rate=0.01, num_epochs=1, data_size=100, batch_size=10)
    # technically we could get very unlucky and have them be the same, but it's very unlikely
    assert not np.all(np.isclose(A1, A2)) 

    # NOTE: randomization should happen *inside* the epoch loop so each epoch is random
    #       ... not clear how to test for this
    #       if randomization happens once outside the loop then learning should be worse...

    # SGD should do better than GD with the same number of passes through the data
    for _ in range(10): # run this multiple times to make sure it's not just a fluke
        A1 = gradient_descent(lambda A: get_linear_regression_gradient(A, X, y), A_init, learning_rate=0.01, num_iterations=100)
        A2 = stochastic_gradient_descent(get_batch_gradient, A_init, learning_rate=0.01, num_epochs=100, data_size=100, batch_size=20)
        assert compute_model_error(x, y, A1, get_simple_linear_features) > compute_model_error(x, y, A2, get_simple_linear_features)

def test_create_sine_data(num_samples = 100, x_range = [-np.pi,np.pi], noise = 0):
    # similar test as linear
    x, y = create_sine_data(num_samples=num_samples, x_range=x_range, noise=noise)
    assert type(x) == np.ndarray and type(y) == np.ndarray
    assert x.shape == (num_samples,1)
    assert y.shape == (num_samples,1)
    assert np.all(x >= x_range[0]) and np.all(x <= x_range[1])
    for i in range(num_samples):
        assert y[i] <= np.sin(x[i]) + noise and y[i] >= np.sin(x[i]) - noise

    # check that the samples are random
    x_new, _ = create_sine_data(num_samples=num_samples, x_range=x_range, noise=noise)
    assert not np.all(np.isclose(x, x_new))

    # check that creating 10,000,000 samples is less than a second
    start = time.time()
    x, y = create_sine_data(num_samples=10000000, x_range=x_range, noise=noise)
    assert time.time() - start < 1

def test_get_polynomial_features(num_data, num_features, degree):
    x = np.random.randn(num_data, num_features)
    X = get_polynomial_features(x, degree=degree)
    assert type(X) == np.ndarray
    assert X.shape == (num_data, num_features * (degree+1))
    for i in range(degree+1):
        assert np.all(np.isclose(X[:, i*num_features : (i+1)*num_features], x**(degree-i)))

def test_ik_loss():
    arm = Arm(np.ones(2))
    q = np.array([0.0,0.0])
    goal_ee = np.array([1.0,1.0])
    loss = ik_loss(arm, q, goal_ee)
    assert loss == np.sqrt(2)
    goal_ee = np.array([2.0,0.0])
    loss = ik_loss(arm, q, goal_ee)
    assert loss == 0.0

def test_sample_near():
    num_samples = 100
    q = np.array([10.0,100.0,300.0])
    epsilon = 1.0
    q_near = sample_near(num_samples, q, epsilon)
    assert type(q_near) == np.ndarray
    assert q_near.shape == (num_samples,q.shape[0])
    assert np.all(np.abs(q_near - q) < epsilon)
    # check that the nearby samples are non-deterministic
    q_near_new = sample_near(num_samples, q, epsilon)
    assert not np.all(np.isclose(q_near, q_near_new))

def test_estimate_ik_gradient():
    arm = Arm(np.ones(4))
    q = np.array([0.0,0.0,0.0,0.0])
    goal_ee = np.array([0.0,5.0])
    loss = lambda q: ik_loss(arm, q, goal_ee)
    num_samples = 100
    grad = estimate_ik_gradient(loss, q, num_samples)
    assert type(grad) == np.ndarray
    assert grad.shape == q.shape
    # test that unit vector
    assert np.isclose(np.linalg.norm(grad), 1.0)
    # check that moving in opposite direction of grad is improving the loss
    assert loss(q - grad) < loss(q)

def test_logistic_loss():
    y_true = np.array([1,0])
    y_pred = np.array([0.9,0.1])
    loss = logistic_loss(y_pred, y_true)
    assert np.isclose(loss, -2*np.log(0.9))

    y_pred = np.array([0.1,0.9])
    loss = logistic_loss(y_pred, y_true)
    assert np.isclose(loss, -2*np.log(0.1))

def test_logistic_error():
    y_true = np.array([1,0])
    y_pred = np.array([0.9,0.1])
    error = logistic_error(y_pred, y_true)
    assert error == 0

    y_pred = np.array([0.1,0.9])
    error = logistic_error(y_pred, y_true)
    assert error == 1

    y_pred = np.array([0.4,0.4])
    error = logistic_error(y_pred, y_true)
    assert np.isclose(error, 0.5)

def test_logistic_prediction():
    # should be 0.5 where y=0, 1 where y is large and 0 where y is small
    # y=0 at x=-3/2
    A = np.array([[2],[3]])
    x = np.array([[-3/2], [-100], [100]])
    y = logistic_prediction(x, A, get_modified_features=get_simple_linear_features)
    assert np.isclose(y[0], 0.5)
    assert np.isclose(y[1], 0)
    assert np.isclose(y[2], 1)

def test_logistic_regression_gradient():
    A = np.array([[2],[-3]])
    x = np.array([[3/2], [-100], [100]])
    X = get_simple_linear_features(x)
    y_true = np.array([[0],[0],[1]])
    grad = get_logistic_regression_gradient(A, X, y_true)
    assert type(grad) == np.ndarray
    assert grad.shape == A.shape
    # grad should be 1/2 * X[0]
    assert np.all(np.isclose(grad, 1/2 * X[0].reshape(-1,1)))

def test_get_logistic_regression_features():
    # needs to get low error on all three datasets
    from main import load_data
    for data_idx in range(3):
        print(f"Testing data {data_idx} GD")
        x,y = load_data(data_idx)
        get_features = get_logistic_regression_features
        X = get_features(x)
        A_init = np.random.rand(X.shape[1], 1)
        # gd
        get_gradient = lambda A: get_logistic_regression_gradient(A, X, y)
        A_gd = gradient_descent(get_gradient, A_init, learning_rate=0.01, num_iterations=10000)
        y_pred = logistic_prediction(x, A_gd, get_features)
        final_error = logistic_error(y_pred, y)
        assert final_error <= 0.1 # 90% accuracy
        
        # sgd
        print(f"Testing data {data_idx} SGD")
        get_batch_gradient = lambda A, indices: get_logistic_regression_gradient(A, X[indices], y[indices])
        A_sgd = stochastic_gradient_descent(get_batch_gradient, A_init, learning_rate=0.001, 
                                            num_epochs=10000, data_size=x.shape[0], batch_size=20)
        y_pred = logistic_prediction(x, A_sgd, get_features)
        final_error = logistic_error(y_pred, y)
        assert final_error <= 0.1 # 90% accuracy

if __name__ == "__main__":
    # you should comment out what you're not testing...
    test_create_linear_data(num_samples=10, slope=2, intercept=-1, x_range=[-10,10], noise=0)
    test_create_linear_data(num_samples=10, slope=-3, intercept=2, x_range=[0,5], noise=0.2)
    test_get_simple_linear_features(num_samples=10, num_features=1)
    test_get_simple_linear_features(num_samples=1, num_features=4)
    test_linear_prediction()
    test_mse_loss()
    test_analytical_linear_regression()
    test_linear_regression_gradient()
    test_gradient_descent()
    test_stochastic_gradient_descent()
    test_create_sine_data()
    test_get_polynomial_features(num_data=10, num_features=1, degree=3)
    test_get_polynomial_features(num_data=10, num_features=2, degree=3)
    test_get_polynomial_features(num_data=100, num_features=1, degree=1)
    test_ik_loss()
    test_sample_near()
    test_estimate_ik_gradient()
    test_logistic_error()
    test_logistic_prediction()
    test_logistic_loss()
    test_logistic_regression_gradient()
    test_get_logistic_regression_features()