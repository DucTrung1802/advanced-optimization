# LINEAR REGRESSION FRAMEWORK

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Theory
# For a record, x is input vector with m features
# Then y0 is a value of output
# x.shape = [m, 1] => 1 record, m features
# y0.shape = [1, 1]

# => X is a matrix of all input records
# Then y is a vector of all output
# X.shape = [m, n] => n records, m features
# y.shape = [n, 1] => n records

# w is a vector of weights for each feature
# w.shape = [m, 1] => m features


# SETUP SAMPLE DATA

NUMBER_OF_RECORDS = 1000
RANGE_OF_NOISE = 0.5
w0 = 4
w1 = 3


# SET UP GRADIENT DESCENT PARAMETERS

INITIAL_W = np.array([2, 1])
LEARNING_RATE = 1

# =====================================================================

# TRANSFORMATIONS
X = np.random.rand(NUMBER_OF_RECORDS).reshape(1, -1)

y = w0 + X.T * w1
NOISE = RANGE_OF_NOISE * np.random.randn(NUMBER_OF_RECORDS).reshape(
    NUMBER_OF_RECORDS, 1
)
y += NOISE

# Build 6: add bias for X
one = np.ones((1, X.shape[1]))
Xbar = np.concatenate((one, X), axis=0)


# CLASSES


class LinearModel:
    def __init__(self, b, w, X_plot, y_plot):
        self.b = b
        self.w = w
        self.X_plot = X_plot
        self.y_plot = y_plot


# CORE FUNCTIONS


def cost(w: np.ndarray, Xbar: np.ndarray, y: np.ndarray):
    # w is a vector that included the bias
    w = w.reshape(-1, 1)
    N = Xbar.shape[1]
    return (
        0.5 / N * np.linalg.norm(y.reshape(-1, 1) - Xbar.T.dot(w).reshape(-1, 1)) ** 2
    )


def grad(w: np.float64, Xbar: np.ndarray, y: np.ndarray):
    # w is a vector that included the bias
    w = w.reshape(-1, 1)
    N = Xbar.shape[1]
    return 1 / N * Xbar.dot(Xbar.T.dot(w).reshape(-1, 1) - y.reshape(-1, 1))


def check_grad(fn, gr, X, Xbar, y):
    # convert X to an 1d array, later we'll need only one for loop
    X_flat = X.reshape(-1)
    shape_X = X.shape  # original shape of X
    num_grad = np.zeros_like(X)  # numerical grad, shape = shape of X
    grad_flat = np.zeros_like(X_flat)  # 1d version of grad
    eps = 1e-6  # a small number, 1e-10 -> 1e-6 is often good
    numElems = X_flat.shape[0]  # number of elements in X
    # calculate numerical gradient
    for i in range(numElems):  # iterate over all elements of X
        Xp_flat = X_flat.copy()
        Xn_flat = X_flat.copy()
        Xp_flat[i] += eps
        Xn_flat[i] -= eps
        Xp = Xp_flat.reshape(shape_X)
        Xn = Xn_flat.reshape(shape_X)
        fp_ = fn(Xp, Xbar, y)
        fn_ = fn(Xn, Xbar, y)
        grad_flat[i] = (fp_ - fn_) / (2 * eps)

    num_grad = grad_flat.reshape(shape_X)

    diff = np.linalg.norm(num_grad.reshape(-1, 1) - gr(X, Xbar, y).reshape(-1, 1))
    print("Difference between two methods should be small:", diff)


def iterateGD(grad, w0, learning_rate, Xbar, y):
    # w0 is a vector that included the bias
    w0 = w0.reshape(-1, 1)
    w = [w0]
    for it in range(100):
        w_new = w[-1] - learning_rate * grad(w[-1], Xbar, y)
        if np.linalg.norm(grad(w_new, Xbar, y)) / np.array(w0).size < 1e-3:
            break
        w.append(w_new)
    return w


def visualize(X, y, model: LinearModel = None, label=None):
    plt.scatter(X, y, color="black", label="Data points")

    if model:

        # Display the model equation
        plt.text(
            0.05,
            0.95,
            f"Model: f(x) = {model.b:.3f} + {model.w:.3f}x",
            transform=plt.gca().transAxes,
        )

        # Predict using the model
        plt.plot(
            model.X_plot,
            model.y_plot,
            color="red",
            label=label,
            linewidth=4,
        )

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()


def main():
    global X, y, NOISE
    # Visualize sample data points
    y += NOISE
    visualize(X, y)

    # Create model using sklearn
    model = LinearRegression()
    model.fit(X.T, y)

    # Extract the coefficients
    b = model.intercept_[0]
    w = model.coef_[0][0]
    X_plot = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    sklearn_model = LinearModel(b, w, X_plot, y_plot)

    visualize(X, y, sklearn_model, "sklearn LR model")

    # Find solution using GD

    # 1. Check numerical gradient
    check_grad(cost, grad, np.random.randn(2), Xbar, y)

    # 2. Calculate weights after each iteration
    w1 = iterateGD(grad, INITIAL_W, LEARNING_RATE, Xbar, y)

    # 3. Visualize the process of GD


if __name__ == "__main__":
    main()
