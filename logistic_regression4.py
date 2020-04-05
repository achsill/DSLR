import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    return result[0: 800], result[800: 1600]

def calculate_z(X, W, b):
    tmp_X0 = []
    tmp_X1 = []
    result = np.array([])
    i = 0
    for elem in X:
        tmp_X0.append(elem[0] * W[0])
        tmp_X1.append(elem[0] * W[1])

    while i < len(tmp_X0):
        result = np.append(result, b + tmp_X0[i] + tmp_X1[i])
        i+=1


    # print(result)
    # print(b, X[0], W[0], X[1], W[1])
    return result

def cost_func(y, y_hat):
    return -1/m  * sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

df = pd.read_csv('ressources/dataset_train.csv')
X = df[["Herbology", "Astronomy", "Hogwarts House", "Ancient Runes"]]
X = X.dropna()

X = X.fillna(X.mean())
y = X[["Hogwarts House"]]
# print(y)
X = X[["Herbology", "Ancient Runes", "Astronomy"]]

X=((X-X.min())/(X.max()-X.min()))


# print(X[["Herbology"]])
# print(X[["Ancient Runes"]])
X1 = np.array(X[["Herbology"]])
X2 = np.array(X[["Ancient Runes"]])
X3 = np.array(X[["Astronomy"]])

# print(len(X1))
print("Les len: ", str(len(X)) + " " + str(len(X1)) + " ", str(len(X1)) + " ",str(len(y)) + " ")

X1_train, X1_test = X1[0: 800], X1[800: 1600]
X2_train, X2_test = X2[0: 800], X2[800: 1600]
X3_train, X3_test = X3[0: 800], X3[800: 1600]
x_train, x_test = np.concatenate((X1_train, X2_train, X3_train), axis=1), np.concatenate((X1_test, X2_test, X3_test), axis=1)

# perm = np.random.permutation(150)
# x_train, x_test = X[perm][:20], X[perm][:20]
# y_train, y_test = y[perm][:20], y[perm][:20]
#
# test1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# test2 = np.array([1, 2, 3, 4])

# print(X1_train[0], X2_train[0])
# print(x_train)

learning_rate = 0.1



m=len(y)

# def calculate_dw(X, S, y):
#     result = 0
#     X = np.transpose(X)
#     i = 0
#
#     while i < len(y):
#         result = result + ((S-y) * X[0][i])
#         i+=1
#
#     i = 0
#     while i < len(y):
#         result = result + ((S-y) * X[1][i])
#         i+=1
#
#     return result
#
def predict(y_predict):
    result = np.array([])
    for y in y_predict:
        if y >= 0.5:
            result = np.append(result, 1)
        else:
            result = np.append(result, 0)
    print(result)
    return result


results_done = []
houses = ["Slytherin", "Hufflepuff", "Gryffindor", "Ravenclaw"]
for house in houses:
    y_train, y_test = change_house_to_number(list(y.iloc[:, 0]), house)
    y_train = y_train.reshape(-1, 1)
    W = np.zeros((3, 1))
    b = np.zeros((1, 1))
    tmp_cost = 0
    tmp = []
    for epoch in range(50000):
        Z = np.matmul(x_train, W) +  b
        # print("__________")
        # print(Z)
        # print("__________")
        S = sigmoid(Z)
        cost = cost_func(y_train, S)

        if (abs(tmp_cost == cost)):
            print("haha")
            break

        # print(S)
        # dw = 1/m * sum(np.matmul(x_train.T[0], S - y_train))
        db = 1/m * sum((S - y_train))
        W[0] = W[0] - learning_rate * (1/m * sum(np.matmul(x_train.T[0], S - y_train)))
        W[1] = W[1] - learning_rate * (1/m * sum(np.matmul(x_train.T[1], S - y_train)))
        W[2] = W[2] - learning_rate * (1/m * sum(np.matmul(x_train.T[2], S - y_train)))
        # W = W - learning_rate * dw
        b = b - learning_rate * db
        # print("W")
        # print(W)
        # print(b)
        # print("W")
        tmp_cost = cost

    # print("_______________")
    # print(W), print(b)
    # print("_______________")
    tmp.append(W[0][0])
    tmp.append(W[1][0])
    tmp.append(W[2][0])
    tmp.append(b[0][0])
    results_done.append(tmp)
    y_result = predict(sigmoid(np.matmul(x_test, W) +  b))
    print("House: " , house)
    print(accuracy_score(y_result, y_test))
    print(W, b)



# print(W)
# print(b)
print(results_done)

# test1 = np.concatenate((x_test, y_test), axis=1)
# print(test1)

# print(sigmoid(calculate_z(3,W,b)))
#

def get_multi(theta_list):
    result = []
    for elem in x_test:
        tmp = []
        for theta in theta_list:
            np.matmul(x_test, W) +  b
            tmp.append(sigmoid((elem[0] * theta[0]) + (elem[1] * theta[1]) + (elem[2] * theta[2]) + theta[3]))
        print(tmp)
        maison = tmp.index(max(tmp))
        result.append(houses[maison])
    return result


enfin = get_multi(results_done)
print("______")
print(len(x_train), len(y_train))
y_result = predict(sigmoid(np.matmul(x_test, W) +  b))
la1 = y.iloc[800: 1600]
la2 = la1["Hogwarts House"].to_list()

print(accuracy_score(la2, enfin))
plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train.ravel())
ax = plt.gca()
xvals = np.array(ax.get_xlim()).reshape(-1, 1)
yvals = - (xvals * W[0][0] + b) / W[1][0]
plt.plot(xvals, yvals)
# plt.scatter(x_train, y)
# plt.plot(X, sigmoid(np.matmul(x_test, W) +  b))
# plt.show()


# 0.9768707482993197
