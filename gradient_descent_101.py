# Function to be optimized
# f(x) = x^2 + 5*sin(x)

import math


def cost(x):
    return x**2 + 5 * math.sin(x)


def grad(x):
    return 2 * x + 5 * math.cos(x)


def myGD(x0, eta):
    x = [x0]
    for iter in range(100):
        xNew = x[-1] - eta * grad(x[-1])
        if abs(grad(xNew)) < 1e-3:
            break
        x.append(xNew)
    return (x, iter)


def main():
    x0 = 4
    eta = 0.1
    (x1, it1) = myGD(x0, eta)
    print(
        "Solution x1 = %f, cost = %f, after %d iterations" % (x1[-1], cost(x1[-1]), it1)
    )


if __name__ == "__main__":
    main()
