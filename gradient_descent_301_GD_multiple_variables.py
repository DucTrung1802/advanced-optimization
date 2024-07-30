import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.random.rand(1000)
y = 4 + 3 * X + 0.5 * np.random.randn(1000)  # noise added

# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X.reshape(-1, 1)), axis=1)

model = LinearRegression()
model.fit(X.reshape(-1, 1), y.reshape(-1, 1))

w, b = model.coef_[0][0], model.intercept_[0]
sol_sklearn = np.array([b, w])
print("Solution found by sklearn:", sol_sklearn)


def cost(w):
    N = Xbar.shape[0]
    alo = y - Xbar.dot(w)
    abc = np.linalg.norm(y - Xbar.dot(w))
    return 0.5 / N * np.linalg.norm(y - Xbar.dot(w)) ** 2


def grad(w):
    N = Xbar.shape[0]
    return 1 / N * Xbar.T.dot(Xbar.dot(w) - y)


def GD(grad, x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if np.linalg.norm(grad(x_new)) / np.array(x0).size < 1e-3:
            break
        x.append(x_new)
    return x


def check_grad(fn, gr, X):
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
        fp_ = fn(Xp)
        fn_ = fn(Xn)
        grad_flat[i] = (fp_ - fn_) / (2 * eps)

    num_grad = grad_flat.reshape(shape_X)

    diff = np.linalg.norm(num_grad - gr(X))
    print("Difference between two methods should be small:", diff)


def compute_ellipse_params(X, y):
    N = X.shape[0]
    a1 = np.linalg.norm(y, 2) ** 2 / N
    b1 = 2 * np.sum(X) / N
    c1 = np.linalg.norm(X, 2) ** 2 / N
    d1 = -2 * np.sum(y) / N
    e1 = -2 * X.T.dot(y) / N
    return a1, b1, c1, d1, e1


def generate_grid(delta=0.025):
    xg = np.arange(1.5, 6.0, delta)
    yg = np.arange(0.5, 4.5, delta)
    Xg, Yg = np.meshgrid(xg, yg)
    return Xg, Yg


def compute_Z(Xg, Yg, a1, b1, c1, d1, e1):
    Z = a1 + Xg**2 + b1 * Xg * Yg + c1 * Yg**2 + d1 * Xg + e1 * Yg
    return Z


def lr_gd_draw(w1, filename, Xg, Yg, Z, b, w):
    w_hist = np.array(w1)
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.cla()
    plt.axis([1.5, 6, 0.5, 4.5])
    plt.tick_params(axis="both", which="major", labelsize=13)
    CS = plt.contour(Xg, Yg, Z, 100)
    plt.plot(
        w_hist[:, 0],
        w_hist[:, 1],
        marker="o",
        color="r",
        linestyle="-",
        markeredgecolor="k",
    )
    plt.plot(b, w, "go")
    plt.plot(w_hist[0, 0], w_hist[0, 1], "bo")
    str0 = "%d iterations" % w_hist.shape[0]
    plt.title(str0)
    plt.savefig(filename)
    plt.show()


def main():
    check_grad(cost, grad, np.random.randn(2))
    w_init = np.array([2, 1])
    w1 = GD(grad, w_init, 1)
    print("Sol found by GD: w = ", w1[-1], ",\nafter %d iterations." % len(w1))

    a1, b1, c1, d1, e1 = compute_ellipse_params(X, y)
    Xg, Yg = generate_grid()
    Z = compute_Z(Xg, Yg, a1, b1, c1, d1, e1)

    lr_gd_draw(w1, "LR_gd_1.pdf", Xg, Yg, Z, b, w)


if __name__ == "__main__":
    main()
