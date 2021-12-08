
import numpy
import pandas
import matplotlib.pyplot as plt


def hypothesis(theta, x):
    return theta + (theta * x)


def cost_function(x, y, theta):
    m = len(x)  # number of training examples
    sum = 0
    for i in range(m):
        sum += (hypothesis(theta, x[i]) - y[i]) ** 2
    J = (1 / (2 * m)) * sum
    return J


def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)  # number of training examples
    for i in range(iterations):
        sum = 0
        for j in range(m):
            sum += (hypothesis(theta, x[j]) - y[j])
        theta = theta - (alpha/m) * sum
        cost = cost_function(x, y, theta)
        if i % 10 == 0:  # just look at cost every ten loops for debugging
            print("theta: ", theta, " cost: ", cost)
    return theta


def predict(x, theta):
    y_pred = []
    for i in range(len(x)):
        y_pred.append(hypothesis(x[i], theta))
    return y_pred


if __name__ == '__main__':

    # Read data
    url = "house_data.csv"
    dataset = pandas.read_csv(url,index_col=0)

    # Get the X and Y values
    x = dataset['sqft_living'].values
    y = dataset['price'].values

    # split dataset into training and testing data
    size = round(len(dataset) / 3)
    x_train = x[:size]
    y_train = y[:size]

    x_test = x[size:]
    y_test = y[size:]

    theta = 0

    print(cost_function(x_train, y_train, theta))

    alpha = 0.0003
    iterations = 500

    theta = gradient_descent(x_train, y_train, theta, alpha, iterations)
    y_pred = predict(x_test, theta)

    # Plot regression line across data points
    plt.xlabel("Sqft_living")
    plt.ylabel("Price")
    plt.title("Real vs predicted values")
    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_pred)
    plt.show()
