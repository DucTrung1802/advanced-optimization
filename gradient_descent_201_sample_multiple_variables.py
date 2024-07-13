import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def visualize(X, y, model):
    plt.scatter(X, y, color="blue", label="Data points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Fitted line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()


def main():
    # Generate random data
    X = np.random.rand(1000)
    y = 4 + 3 * X + 0.5 * np.random.randn(1000)  # noise added

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))

    # Extract the coefficients
    w, b = model.coef_[0][0], model.intercept_[0]
    sol_sklearn = np.array([b, w])
    print(sol_sklearn)

    # Visualize the data and the regression line
    visualize(X, y, model)


if __name__ == "__main__":
    main()
