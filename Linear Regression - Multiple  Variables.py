import numpy
import pandas
import matplotlib.pyplot as plt


def hypothesis(theta, x, index):
    new_x = numpy.append([1], x[index])
    result = new_x @ theta
    return result


def cost_function(x, y, theta):
    m = len(x)  # number of training examples
    sum = 0
    for i in range(m):
        sum += numpy.power((hypothesis(theta, x, i) - y[i]), 2)
    J = (1 / (2 * m)) * sum
    return J


def gradient_descent(x, y, theta, alpha, iterations):
    m = len(x)  # number of training examples
    iter = []
    costs = []
    for iteration in range(iterations):
        # Calculate theta
        theta0 = False
        newTheta = []
        sum_theta_zero = 0
        sum_all = 0
        # loop over the features to calculate the theta for each feature
        for j in range(len(x[0])):
            for i in range(m):
                sum_all += (hypothesis(theta, x, i) - y[i]) * x[i][j]
                if (not theta0):
                    sum_theta_zero += (hypothesis(theta, x, i) - y[i])

            if (not theta0):
                newTheta.append(theta[0] - (alpha / m) * sum_theta_zero)
                newTheta.append(theta[1] - (alpha / m) * sum_all)
                theta0 = True
            else:
                newTheta.append(theta[j+1] - (alpha / m) * sum_all)

        theta = newTheta
        # Calculate the cost
        cost = cost_function(x, y, theta)
        costs.append(cost)
        iter.append(iteration)
        if iteration % 10 == 0:  # just look at cost every ten loops for debugging
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
        y_pred.append(hypothesis(theta, x, i))
    return y_pred


if __name__ == '__main__':

    # Read data
    url = "house_data.csv"
    dataset = pandas.read_csv(url,index_col=0)

    # Feature Scaling, Mean Normalization
    dataset = (dataset - dataset.mean()) / dataset.std()

    # Get the X and Y values
    x = []
    x.append(dataset['sqft_living'].values)
    x.append(dataset['grade'].values)
    x.append(dataset['bathrooms'].values)
    x.append(dataset['lat'].values)
    x.append(dataset['view'].values)
    y = dataset['price'].values

    y = numpy.array(y)
    x = numpy.array(x)
    x = x.T

    # split dataset into training and testing data
    size = round(len(dataset) * 0.7)
    x_train = x[:size]
    x_test = x[size:]

    y_train = y[:size]
    y_test = y[size:]

    theta = [0, 0, 0, 0, 0, 0]

    print(cost_function(x_train, y_train, theta))

    alpha = 0.2
    iterations = 100

    theta = gradient_descent(x_train, y_train, theta, alpha, iterations)
    print(theta)
    y_pred = predict(x_test, theta)

    # Plot predicated y and actual y
    plt.title("Real vs predicted values")
    plt.scatter(x=list(range(0, len(y_test))), y=y_test, color='blue')
    plt.scatter(x=list(range(0, len(y_pred))), y=y_pred, color='red')

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
