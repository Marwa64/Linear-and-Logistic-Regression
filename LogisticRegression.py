import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# dimension is the number of features
def weightInitialization(dimension):
    w = np.zeros((1, dimension))
    b = 0
    return w, b


def sigmoid_function(z):
    s = 1 / (1 + np.exp(-z))
    return s


# this is to that ensure there is a forward propagation and at the same time, returns the cost
def gradient_descent(X, Y, w, b):
    # Get the number of simple examples
    m = X.shape[0]

    # Prediction
    A = sigmoid_function(np.dot(w, X.T) + b)
    Y_T = Y.T

    # cost function
    cost = (-1 / m) * (np.sum((Y_T * np.log(A)) + ((1 - Y_T) * (np.log(1 - A)))))

    # Gradient calculation
    dw = (1 / m) * (np.dot(X.T, (A - Y.T).T))  # this is derivative of the cost function with respect to w
    db = (1 / m) * (np.sum(A - Y.T))  # this is derivative of the cost function with respect to b

    grads = {"dw": dw, "db": db}

    return grads, cost


# We are trying to get the parameters w and b after modifying them using the knowledge of the cost function
def model(X, Y, w, b, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        grads, cost = gradient_descent(X, Y, w, b)

        dw = grads["dw"]
        db = grads["db"]
        # parameters update
        w = w - (learning_rate * dw.T)
        b = b - (learning_rate * db)

        costs.append(cost)
        if i % 100 == 0:
            print("Cost after %i iteration is %f" % (i, cost))

    # final parameters
    coefficients = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}

    return coefficients, gradient, costs


def predict(final_predicted, m):
    y_predicted = np.zeros((1, m))
    for i in range(final_predicted.shape[1]):
        if final_predicted[0][i] > 0.5:
            y_predicted[0][i] = 1
    return y_predicted


if __name__ == '__main__':
    df = pd.read_csv('heart.csv')

    X = df.drop(columns=["age", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"])
    Y = df["target"]

    # Scalar applies dimensionality reduction on X to reduce the time and storage space required
    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    '''
    random_state is basically used for reproducing your problem the same every time it is run. 
    If you do not use a random_state in train_test_split, every time you make the split 
    you might get a different set of train and test data points.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    X_tr_arr = X_train
    X_ts_arr = X_test
    y_tr_arr = y_train.to_numpy()
    y_ts_arr = y_test.to_numpy()

    print('The shape of the input training set is {}'.format(X_train.shape))
    print('The shape of the output training set is {}'.format(y_train.shape))
    print('The shape of the input testing set is {}'.format(X_test.shape))
    print('The shape of the output testing set is {}'.format(y_test.shape))

    # Get number of features
    number_features = X_tr_arr.shape[1]
    w, b = weightInitialization(number_features)
    # Gradient Descent
    coeff, gradient, costs = model(X_tr_arr, y_tr_arr, w, b, learning_rate=0.0001, no_iterations=50000)

    # Final prediction
    w = coeff["w"]
    b = coeff["b"]
    # print('Optimized thetas', w)
    # print('Optimized intercept', b)

    final_train_predicted = sigmoid_function(np.dot(w, X_tr_arr.T) + b)
    final_test_predicted = sigmoid_function(np.dot(w, X_ts_arr.T) + b)
    #
    m_tr = X_tr_arr.shape[0]
    m_ts = X_ts_arr.shape[0]

    y_tr_predicted = predict(final_train_predicted, m_tr)
    print('Training Accuracy {:.2f}%'.format(100 * accuracy_score(y_tr_predicted.T, y_tr_arr)))

    y_ts_predicted = predict(final_test_predicted, m_ts)
    print('Test Accuracy {:.2f}%'.format(100 * accuracy_score(y_ts_predicted.T, y_ts_arr)))