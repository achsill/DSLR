import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def read_file():
    df = pd.read_csv('ressources/dataset_train.csv')
    X = df[["Herbology", "Astronomy", "Hogwarts House", "Ancient Runes", "Defense Against the Dark Arts"]]
    X = X.dropna()
    # X = X.fillna(X.mean())
    y = X[["Hogwarts House"]]
    X = X[["Herbology", "Ancient Runes", "Astronomy", "Defense Against the Dark Arts"]]
    # Herbology = X[["Herbology"]]
    # print(Herbology.isna().sum())
    # Ancient = X[["Ancient Runes"]]
    # print(Ancient.isna().sum())
    # Astronomy = X[["Astronomy"]]
    # print(Astronomy.isna().sum())
    # Def = X[["Defense Against the Dark Arts"]]
    # print(Def.isna().sum())
    X = ((X-X.min())/(X.max()-X.min()))
    X = X.to_numpy()
    return X[0: 400], X[400: 1600], y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def change_house_to_number(y, house_name):
    result = np.array([])
    for house in y:
        if house == house_name:
            result = np.append(result, 1)
        else:
            result = np.append(result, 0)
    result = np.array(result)
    return result[0: 400], result[400: 1600]

def cost_func(y, y_hat):
    return -1/m  * sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

x_train, x_test, y = read_file()
learning_rate = 0.1
m=len(y)

def predict(y_predict):
    result = np.array([])
    for y in y_predict:
        if y >= 0.5:
            result = np.append(result, 1)
        else:
            result = np.append(result, 0)
    return result


results_done = []
houses = ["Slytherin", "Hufflepuff", "Gryffindor", "Ravenclaw"]
for house in houses:
    y_train, y_test = change_house_to_number(list(y.iloc[:, 0]), house)
    y_train = y_train.reshape(-1, 1)
    W = np.zeros((4, 1))
    b = 0
    tmp_cost = 0
    tmp = []
    for epoch in range(60000):
        Z = np.matmul(x_train, W) +  b
        S = sigmoid(Z)
        cost = cost_func(y_train, S)

        # if (abs(tmp_cost - cost) < 0.000001):
        #     break

        db = 1/m * sum((S - y_train))
        for key, weight in enumerate(W):
            W[key] = W[key] - learning_rate * (1/m * sum(np.matmul(x_train.T[key], S - y_train)))
        b = b - learning_rate * db
        tmp_cost = cost

    tmp.append(W[0][0])
    tmp.append(W[1][0])
    tmp.append(W[2][0])
    tmp.append(W[3][0])
    tmp.append(b)
    results_done.append(tmp)
    y_result = predict(sigmoid(np.matmul(x_test, W) +  b))

def get_multi(theta_list):
    result = []
    for elem in x_test:
        tmp = []
        for theta in theta_list:
            np.matmul(x_test, W) +  b
            tmp.append(sigmoid((elem[0] * theta[0]) + (elem[1] * theta[1]) + (elem[2] * theta[2]) + (elem[3] * theta[3]) + theta[4]))
        maison = tmp.index(max(tmp))
        result.append(houses[maison])
    return result

def print_accuracy():
    enfin = get_multi(results_done)
    la1 = y.iloc[400: 1600]
    la2 = la1["Hogwarts House"].to_list()
    print(accuracy_score(la2, enfin))

print_accuracy()
