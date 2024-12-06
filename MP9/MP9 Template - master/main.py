import numpy as np, math
import matplotlib.pyplot as plt
import argparse
from mp9 import *

def run_optimization_and_save_intermediates(optim, A_init, total_iters, plot_iters):
    A_vals = [A_init]
    for _ in range(math.ceil(total_iters / plot_iters)):
        # run a few iterations of optimization and save parameter value
        cur_iters = min(plot_iters, total_iters - len(A_vals)*plot_iters)
        A_vals.append(optim(A_init=A_vals[-1], num_iterations=cur_iters))
    return A_vals

def plot_regression_data(x, y, x_range):
    plt.scatter(x, y, alpha=0.1, color="blue", label="Data")
    plt.xlim(x_range)
    plt.ylim([np.min(y)-1, np.max(y)+1])

def plot_classification_data(x, y):
    plt.scatter(x[y[:,0] == 0,0], x[y[:,0] == 0,1], color='red', label="Class 0")
    plt.scatter(x[y[:,0] == 1,0], x[y[:,0] == 1,1], color='blue', label="Class 1")

def plot_logistic_regression_decision_boundary(A, x, get_features):
    # only use x to determine range of values
    x1 = np.linspace(np.min(x[:,0]) - 1, np.max(x[:,0]) + 1,1000)
    x2 = np.linspace(np.min(x[:,1]) - 1, np.max(x[:,1]) + 1,1000)
    X1, X2 = np.meshgrid(x1, x2)
    X_all = np.concatenate([X1.reshape(-1,1), X2.reshape(-1,1)], axis=1)
    y_pred = logistic_prediction(X_all, A, get_features) > 0.5
    plt.contourf(X1, X2, y_pred.reshape(1000,1000), alpha=0.5)

def load_data(data_id):
    data = np.load(f"data/data_{data_id}.npz")
    x = data['x']
    y = data['y']
    return x, y

# This main function does the following:
#   1. x, y = load or create data
#   2. X = get_problem_specific_modified_features(x)
#   3. A_init = initial parameter values
#   4. get_gradient = lambda A: get_problem_specific_gradient(A, X, y)
#   5. optim = analytical, gd, or sgd (depends on get_gradient)
#   6. A_vals = run_optimization_and_save_intermediates(
#           optim, A_init, args.num_iterations, args.plot_iterations)
#   7. plot the results
def main(args):
    print("Running with args:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print("Please note that some of the arguments may be ignored depending on the data type and method")

    # set the seed if one is provided for reproducible results
    if args.random_seed is not None:
        print(f"Setting random seed to {args.random_seed}")
        np.random.seed(args.random_seed)
    
    # 1. create the data and define data specific functions
    #   - create or load the data
    #   - set get_features for linear, sine, and logistic
    #   - set get_gradient for ik and gradient type for linear, sine, and logistic
    #   - set value for A_init
    get_gradient = None
    if args.data_type in ["linear", "sine", "logistic"]:    
        if args.data_type == "linear":
            x,y = create_linear_data(args.num_samples, args.slope, args.intercept, args.x_range, args.noise)
            get_features = get_simple_linear_features
            gradient_type = get_linear_regression_gradient
            plot_regression_data(x, y, args.x_range)
        elif args.data_type == "sine":
            x,y = create_sine_data(args.num_samples, args.x_range, args.noise)
            get_features = lambda x: get_polynomial_features(x, args.degree)
            gradient_type = get_linear_regression_gradient
            plot_regression_data(x, y, args.x_range)
        elif args.data_type == "logistic":
            if args.method == "analytical":
                raise ValueError("Analytical method not supported for logistic regression")
            x,y = load_data(args.logistic_data_id)
            get_features = get_logistic_regression_features
            gradient_type = get_logistic_regression_gradient
            plot_classification_data(x, y)

        # get modified features
        X = get_features(x)
        # initialize parameters
        A_init = np.random.rand(X.shape[1], 1)

    elif args.data_type == "ik":
        # ik only works for gd
        if args.method != "gd":
            raise ValueError("IK only works with gradient descent")
        
        # initialize the arm and set the goal
        arm_lengths = np.ones(args.num_joints)
        arm = Arm(arm_lengths=arm_lengths)
        arm.draw_space(plt.gca())
        ee_goal = np.array(args.ee_goal)
        # define the loss function and gradient for IK
        ee_goal_loss = lambda c: ik_loss(arm, c, ee_goal)
        get_gradient = lambda c_: estimate_ik_gradient(ee_goal_loss, c_, num_samples=args.batch_size)
        # plot goal position
        plt.scatter(ee_goal[0], ee_goal[1], s=200, color='black', label="Goal")
        # initialize parameters (initial arm position) and plot initial arm position
        A_init = np.random.rand(args.num_joints) * 180 - 90
        arm.draw_config(plt.gca(), A_init, color='blue', label="Initial")
    else:
        raise ValueError(f"Unknown data type: {args.data_type}")

    # 2. set optimization method and get_gradient function
    # The only difference between gd and sgd is that get_gradient should accept indices for sgd
    # Analytical method only works for linear regression (not logistic or IK)
    if args.method == "analytical":
        A_vals = [analytical_linear_regression(X, y)]
        optim = None
    elif args.method == "gd":
        if get_gradient is None:
            get_gradient = lambda A: gradient_type(A, X, y)
        optim = lambda A_init, num_iterations: gradient_descent(
            get_gradient=get_gradient,
            A_init=A_init, 
            learning_rate=args.learning_rate, 
            num_iterations=num_iterations)
    elif args.method == "sgd":
        if get_gradient is None:
            get_gradient = lambda A, indices: gradient_type(A, X[indices], y[indices])
        optim = lambda A_init, num_iterations: stochastic_gradient_descent(
            get_batch_gradient = get_gradient,
            A_init = A_init,
            learning_rate = args.learning_rate, 
            num_epochs = num_iterations, 
            data_size = X.shape[0],
            batch_size = args.batch_size)
    else:
        raise ValueError(f"Unknown optimization method: {args.method}")
    
    # 3. run optimization (either gd or sgd) and plot the results
    print(f"Running {args.method} optimization for {args.num_iterations} iterations")
    if optim is not None:
        A_vals = run_optimization_and_save_intermediates(optim, A_init, args.num_iterations, args.plot_iterations)
    
    # 4. plot the results
    x_plot = np.linspace(args.x_range[0], args.x_range[1], 1000).reshape(-1, 1)
    model_errors = []
    for i, A_val in enumerate(A_vals):
        plot_params = {"color": 'red', "alpha": 0.2}
        if i == 0:
            plot_params["label"] = "Intermediate"
        if i == len(A_vals) - 1:
            plot_params = {"color": 'green', "label": "Final"}
            if args.data_type == "logistic":
                plot_logistic_regression_decision_boundary(A_val, x, get_features)
        
        if args.data_type == "ik":
            arm.draw_config(plt.gca(), A_val, **plot_params)
            model_errors.append(ee_goal_loss(A_val))
        elif args.data_type == "logistic":
            y_pred = logistic_prediction(x, A_val, get_features)
            model_errors.append(logistic_error(y_pred, y))
        else:
            y_plot = linear_prediction(x_plot, A_val, get_features)
            plt.plot(x_plot, y_plot, linewidth=3.0, **plot_params)
            model_errors.append(compute_model_error(x, y, A_val, get_features))
    model_errors = np.array(model_errors)
    
    plt.legend()
    if args.save_path is not None:
        plt.savefig(f"{args.save_path}.png")
        plt.clf()
    else:
        plt.show()
    if len(model_errors) > 1:
        # plot the model errors
        print(f"Plotting model errors: {model_errors}")
        plt.plot(np.arange(len(model_errors)) * args.plot_iterations, model_errors)
        plt.xlabel("Iterations")
        plt.ylabel("Model Error")
        if args.save_path is not None:
            plt.savefig(f"{args.save_path}_errors.png")
            plt.clf()
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP3 Search')
    parser.add_argument('--data_type',dest="data_type", type=str, default="linear",
                        help='What kind of data to generate: [linear, sine, ik]')
    parser.add_argument('--method',dest="method", type=str, default="analytical",
                        help='Which method to use for regression: [analytical, gd, sgd')
    parser.add_argument('--random_seed',dest="random_seed", type=int, default=None,
                        help='Random seed for data generation')
    parser.add_argument('--plot_iterations',dest="plot_iterations", type=int, default=5,
                        help='Number of iterations after which to plot arm for IK')
    parser.add_argument('--save_path',dest="save_path", type=str, default=None,
                        help='Save plots instead of showing them')
    # data creation args
    parser.add_argument('--num_samples',dest="num_samples", type=int, default=128,
                        help='Number of samples to generate')
    parser.add_argument('--slope',dest="slope", type=float, default=2.0,
                        help='Slope of the linear data')
    parser.add_argument('--intercept',dest="intercept", type=float, default=3.0,
                        help='Intercept of the linear data')
    parser.add_argument('--x_range',dest="x_range", type=float, nargs=2, default=[-1.0, 1.0],
                        help='Range of x values')
    parser.add_argument('--noise',dest="noise", type=float, default=0.1,
                        help='Noise in the data')
    # GD and SGD args
    parser.add_argument('--learning_rate',dest="learning_rate", type=float, default=0.1,
                        help='Learning rate for gradient descent')
    parser.add_argument('--num_iterations',dest="num_iterations", type=int, default=10,
                        help='Number of iterations for GD or number of epochs for SGD')
    parser.add_argument('--batch_size',dest="batch_size", type=int, default=16,
                        help='Batch size for SGD')
    # polynomial regression args
    parser.add_argument('--degree',dest="degree", type=int, default=5,
                        help='Degree of the polynomial in polynomial regression')
    # IK args
    parser.add_argument('--ee_goal',dest="ee_goal", type=float, nargs=2, default=[2.0, 2.0],
                        help='End effector goal for IK')
    parser.add_argument('--num_joints',dest="num_joints", type=int, default=4,
                        help='Number of joints in the arm')
    # logistic regression args
    parser.add_argument('--logistic_data_id',dest="logistic_data_id", type=int, default=0,
                        help='ID of the logistic regression data')
    
    args = parser.parse_args()
    np.set_printoptions(precision=4) # for nicer printing...
    
    if args.num_iterations // args.plot_iterations > 100:
        print("**Warning: You are plotting a lot of intermediate results, this may take a while**")
    main(args)