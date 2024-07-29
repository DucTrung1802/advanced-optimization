# LINEAR REGRESSION FRAMEWORK

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# SETUP SAMPLE DATA

NUMBER_OF_RECORDS = 1000
X = np.random.rand(NUMBER_OF_RECORDS)
y = 4 + 3 * X
NOISE = 0.5 * np.random.randn(NUMBER_OF_RECORDS)


# SET UP GRADIENT DESCENT PARAMETERS


# CORE FUNCTIONS


class LinearModel:
    def __init__(self, b, w, X_plot, y_plot):
        self.b = b
        self.w = w
        self.X_plot = X_plot
        self.y_plot = y_plot


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
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))

    # Extract the coefficients and convert them to float
    b = model.intercept_[0]
    w = model.coef_[0][0]
    X_plot = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    sklearn_model = LinearModel(b, w, X_plot, y_plot)

    visualize(X, y, sklearn_model, "sklearn LR model")


if __name__ == "__main__":
    main()
