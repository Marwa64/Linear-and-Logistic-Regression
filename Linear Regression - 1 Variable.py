import numpy
import pandas
import matplotlib.pyplot as plt


def hypothesis(theta, x):
    return theta[0] + (theta[1] * x)


def cost_function(x, y, theta):
    m = len(x)  # number of training examples
    sum = 0
    for i in range(m):
        sum += numpy.power((hypothesis(theta, x[i]) - y[i]), 2)
    J = (1 / (2 * m)) * sum
    return J


def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)  # number of training examples
    iter = []
    costs = []
    for i in range(iterations):
        # Calculate theta 0
        sum = 0
        for j in range(m):
            sum += (hypothesis(theta, x[j]) - y[j])
        temp0 = theta[0] - (alpha/m) * sum
        # Calculate theta 1
        sum = 0
        for j in range(m):
            sum += ((hypothesis(theta, x[j]) - y[j]) * x[j])
        theta[1] = theta[1] - (alpha / m) * sum
        theta[0] = temp0
        # Calculate the cost
        cost = cost_function(x, y, theta)
        costs.append(cost)
        iter.append(i)
        if i % 10 == 0:  # just look at cost every ten loops for debugging
            print("theta: ", theta, " cost: ", cost)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Learning Rates")
    plt.plot(iter, costs)
    plt.show()
    return theta


def predict(x, theta):
    y_pred = []
    for i in range(len(x)):
        y_pred.append(hypothesis(theta, x[i]))
    return y_pred


if __name__ == '__main__':

    # Read data
    url = "C:/Users/mrawa/Desktop/University/Machine Learning/Assignment1/house_data.csv"
    dataset = pandas.read_csv(url,index_col=0)

    # Get the X and Y values
    x = dataset['sqft_living'].values
    y = dataset['price'].values

    # split dataset into training and testing data
    size = round(len(dataset) * 0.8)
    x_train = x[:size]
    y_train = y[:size]

    x_test = x[size:]
    y_test = y[size:]

    theta = [0, 0]

    print(cost_function(x_train, y_train, theta))

    alpha = 0.00000001
    iterations = 300

    theta = gradient_descent(x_train, y_train, theta, alpha, iterations)
    print(theta)
    y_pred = predict(x_test, theta)

    # Plot regression line across data points
    plt.xlabel("Sqft_living")
    plt.ylabel("Price")
    plt.title("Real vs predicted values")
    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_pred)

    print("-------------------------------------------")
    # calculate errors
    total = 0
    sumOfSquaredDiff = 0
    sum = 0
    for i in range(len(y_test)):
        # calculate error
        err = (y_test[i] - y_pred[i]) ** 2
        total += err
        sum += y_test[i]
    mse = total/len(y_test)
    mean = sum / len(y_test)

    for i in range(len(y_test)):
        temp = (y_test[i] - mean) ** 2
        sumOfSquaredDiff += temp

    r2 = 1 - (total/sumOfSquaredDiff)

    print("Mean Squared Error = ", mse)
    print("R2 Coefficient = ", r2)
    print("Accuracy = ", round((r2 * 100)), "%")

    plt.show()
