import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Function to be optimized
def cost(x):
    return x**2 + 5 * math.sin(x)


def grad(x):
    return 2 * x + 5 * math.cos(x)


def myGD(x0, eta):
    x = [x0]
    for iter in range(100):
        xNew = x[-1] - eta * grad(x[-1])
        if abs(grad(xNew)) < 1e-3:
            x.append(xNew)
            break
        x.append(xNew)
    return x, iter + 1


def main():
    x0 = 7
    eta = 0.1
    x_path, it = myGD(x0, eta)
    print(
        "Solution x1 = %f, cost = %f, after %d iterations"
        % (x_path[-1], cost(x_path[-1]), it)
    )
    return x_path


# Plotting the function and the gradient descent path
x = np.linspace(-7, 7, 400)
y = x**2 + 5 * np.sin(x)

x_path = main()

fig, ax = plt.subplots()
ax.plot(x, y, "b-", label="f(x) = x^2 + 5*sin(x)")
(current_point,) = ax.plot([], [], "ro", label="Current State", markersize=8, zorder=5)
(previous_point,) = ax.plot(
    [], [], "ko", label="Previous State", markersize=8, zorder=4
)
(connection_line,) = ax.plot([], [], "k--", alpha=0.6, zorder=3)

# Adding text annotations for iteration, cost, and gradient
iteration_text = ax.text(0.1, 0.95, "", transform=ax.transAxes)
cost_text = ax.text(0.4, 0.95, "", transform=ax.transAxes)
grad_text = ax.text(0.7, 0.95, "", transform=ax.transAxes)
grad_text = ax.text(0.7, 0.95, "", transform=ax.transAxes)


def init():
    current_point.set_data([], [])
    previous_point.set_data([], [])
    connection_line.set_data([], [])
    iteration_text.set_text("")
    cost_text.set_text("")
    grad_text.set_text("")
    return (
        current_point,
        previous_point,
        connection_line,
        iteration_text,
        cost_text,
        grad_text,
    )


def update(frame):
    if frame > 0:
        current_point.set_data([x_path[frame]], [cost(x_path[frame])])
        previous_point.set_data([x_path[frame - 1]], [cost(x_path[frame - 1])])
        connection_line.set_data(
            [x_path[frame - 1], x_path[frame]],
            [cost(x_path[frame - 1]), cost(x_path[frame])],
        )
    else:
        current_point.set_data([x_path[frame]], [cost(x_path[frame])])
        previous_point.set_data([], [])
        connection_line.set_data([], [])

    iteration_text.set_text(f"Iteration: {frame}/{len(x_path)-1}")
    cost_text.set_text(f"Cost: {cost(x_path[frame]):.4f}")
    grad_text.set_text(f"Grad: {grad(x_path[frame]):.4f}")

    return (
        current_point,
        previous_point,
        connection_line,
        iteration_text,
        cost_text,
        grad_text,
    )


ani = FuncAnimation(
    fig, update, frames=range(len(x_path)), init_func=init, blit=True, repeat=False
)

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()
plt.title("Gradient Descent Optimization")
plt.show()
