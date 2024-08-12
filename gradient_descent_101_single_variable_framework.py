import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# SET UP GRADIENT DESCENT PARAMETERS


def cost(x):
    """Cost function to calculate the cost

    Example:

    f(x) = x**2 + 5 * np.sin(x)

    ∇f(x) = 2 * x + 5 * np.cos(x)

    Args:
        x (float): input

    Returns:
        float: f(x)
    """
    return x**2 + 5 * np.sin(x)


def grad(x):
    """Gradient of the function at x

    Example:

    f(x) = x**2 + 5 * np.sin(x)

    ∇f(x) = 2 * x + 5 * np.cos(x)

    Args:
        x (float): input

    Returns:
        float: ∇f(x)
    """
    return 2 * x + 5 * np.cos(x)


INITIAL_X = 9
LEARNING_RATE = 0.1

GRAPH_START_X = -10
GRAPH_END_X = 10
GRAPH_BINS = 400

GIF_NAME = f"animation_101_lr_{LEARNING_RATE}"
FPS = 5

# CORE FUNCTIONS


def iterateGD(w0=0, eta=0.01, max_iter=200, tolerance=1e-3):
    w = [w0]
    for iter in range(max_iter):
        g = grad(w[-1])
        wNew = w[-1] - eta * g
        if abs(g) < tolerance:
            w.append(wNew)
            break
        w.append(wNew)
    return w, iter + 1


def plotFunctionAndPath(x, y, x_path, gif_file_name="animation_101", fps=5):
    fig, ax = plt.subplots()
    ax.plot(x, y, "b-", label="f(x)")
    (current_point,) = ax.plot(
        [], [], "ro", label="Current State", markersize=8, zorder=5
    )
    (previous_point,) = ax.plot(
        [], [], "ko", label="Previous State", markersize=8, zorder=4
    )
    (connection_line,) = ax.plot([], [], "k--", alpha=0.6, zorder=3)

    iteration_text = ax.text(0.1, 0.95, "", transform=ax.transAxes)
    cost_text = ax.text(0.4, 0.95, "", transform=ax.transAxes)
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
        fig,
        update,
        frames=range(len(x_path)),
        init_func=init,
        blit=True,
        repeat=False,
        interval=200,
    )

    ax.set_xlabel("w")
    ax.set_ylabel("f(x)")
    ax.legend(loc="best")
    plt.title("Gradient Descent Optimization")

    ani.save(gif_file_name + ".gif", writer="imagemagick", fps=fps)
    plt.show()


def main():
    x0 = INITIAL_X
    eta = LEARNING_RATE
    x_path, it = iterateGD(x0, eta)
    print(
        "Solution x1 = %f, cost = %f, after %d iterations"
        % (x_path[-1], cost(x_path[-1]), it)
    )

    x = np.linspace(GRAPH_START_X, GRAPH_END_X, GRAPH_BINS)
    y = cost(x)
    plotFunctionAndPath(x, y, x_path, GIF_NAME, FPS)


if __name__ == "__main__":
    main()
