import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from logreg_train import fill_missing_astronomy_values, read_csv, import_dataframe, gradient_descent
import csv

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_multi(x, theta_list):
    houses = ["Slytherin", "Hufflepuff", "Gryffindor", "Ravenclaw"]
    result = []
    for elem in x:
        tmp = []
        for theta in theta_list:
            predict = 0
            for key, weight in enumerate(theta):
                if (key < len(theta) - 1):
                    predict += theta[key] * elem[key]
                else:
                    predict+= theta[key]
            tmp.append(predict)
        maison = tmp.index(max(tmp))
        result.append(houses[maison])
    return result

def print_accuracy(x, y, W):
    return get_multi(x, W)

def main():
    df = read_csv()
    x, y = import_dataframe(df)
    prediction_result = print_accuracy(x, y, np.load("train_result.npy"))

    with open('houses.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Index", "Hogwarts House"])
        for index, house in enumerate(prediction_result):
            writer.writerow([index, house])

if __name__ == "__main__":
	main()
