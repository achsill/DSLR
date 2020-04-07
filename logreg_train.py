import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import math
import argparse

def read_csv():
    parser = argparse.ArgumentParser()
    parser.add_argument('sourcefile',nargs='*', help='File to parse')
    args = parser.parse_args()
    if (len(args.sourcefile) == 0):
        print("Please select a csv file")
        exit()
    try:
        df = pd.read_csv(args.sourcefile[0])
    except IOError as e:
        print("Wrong file format, please select a correct CSV file")
        exit()
    return df

def isNaN(num):
    return num != num

def fill_missing_astronomy_values(df):
    astro = df.loc[:, "Astronomy"].to_list()
    defense = df.loc[:, "Defense Against the Dark Arts"].to_list()
    result = []
    for key, e in enumerate(defense):
        if isNaN(astro[key]) == True:
            result.append(defense[key] * -100)
        else:
            result.append(astro[key])
    return result

def import_dataframe(df):
    astronomy_without_nan = fill_missing_astronomy_values(df)
    df = df.drop(columns=["Astronomy"])
    df["Astronomy"] = astronomy_without_nan
    # df = df.dropna()
    df = df.fillna(df.mean())
    y = df[["Hogwarts House"]]
    X = df[["Herbology", "Astronomy", "Ancient Runes", "Charms", "Flying", "Transfiguration"]]
    X = ((X-X.min())/(X.max()-X.min()))
    X = X.to_numpy()
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def change_string_to_number(y, house_name):
    result = np.array([])
    for house in y:
        if house == house_name:
            result = np.append(result, 1)
        else:
            result = np.append(result, 0)
    result = np.array(result)
    return result

def cost_func(y, y_hat, m):
    return -1/m  * sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

def predict(y_predict):
    result = np.array([])
    for y in y_predict:
        if y >= 0.5:
            result = np.append(result, 1)
        else:
            result = np.append(result, 0)
    return result

def gradient_descent(dataframe):
    x_train, y = import_dataframe(dataframe)
    learning_rate = 0.1
    m=len(y)
    results_done = []
    houses = ["Slytherin", "Hufflepuff", "Gryffindor", "Ravenclaw"]
    for house in houses:
        y_train = change_string_to_number(list(y.iloc[:, 0]), house)
        y_train = y_train.reshape(-1, 1)
        W = np.zeros((6, 1))
        b = 0
        tmp_cost = 0
        tmp = []
        for epoch in range(60000):
            Z = np.matmul(x_train, W) +  b
            S = sigmoid(Z)
            cost = cost_func(y_train, S, m)

            if (abs(tmp_cost - cost) < 0.00001):
                break

            db = 1/m * sum((S - y_train))
            for key, weight in enumerate(W):
                W[key] = W[key] - learning_rate * (1/m * sum(np.matmul(x_train.T[key], S - y_train)))
            b = b - learning_rate * db
            tmp_cost = cost

        for e in W:
            tmp.append(e[0])
        tmp.append(b)
        results_done.append(tmp)
    return results_done
        # icila = predict(sigmoid(( b + x_test.T[0] * W[0] + x_test.T[1] * W[1] + x_test.T[2] * W[2] + x_test.T[3] * W[3]  + x_test.T[4] * W[4])))
        # print(house + ": " + str(accuracy_score(y_test, icila)))

def main():
    df = read_csv()
    results_done = gradient_descent(df)
    np.save("train_result", results_done)

if __name__ == "__main__":
	main()
