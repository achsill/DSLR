import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from logreg_train import fill_missing_astronomy_values, read_csv, import_dataframe
import csv



from sklearn.linear_model import LogisticRegression
from sklearn import metrics


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
            # tmp.append(sigmoid(theta[6] + elem[0] * theta[0] + elem[1] * theta[1] + elem[2] * theta[2] + elem[3] * theta[3]  + elem[4] * theta[4] + elem[5] * theta[5]))
        maison = tmp.index(max(tmp))
        result.append(houses[maison])
    return result

def print_accuracy(x, y, W):
    enfin = get_multi(x, W)
    print(accuracy_score(y, enfin))
    return enfin

def main():
    df = read_csv()
    x, y = import_dataframe(df)

    logreg = LogisticRegression(solver='lbfgs', multi_class='auto')
    logreg.fit(x, y)

    x = x
    y = y

    y_pred = logreg.predict(x)
    print("le leurs ", accuracy_score(y_pred, y))

    W = np.load("train_result.npy")
    prediction_result = print_accuracy(x, y, W)

    with open('houses.csv', mode='w') as csv_file:
        fieldnames = ['Index', 'House']
        writer = csv.writer(csv_file)
        writer.writerow(["Index", "House"])
        for index, house in enumerate(prediction_result):
            writer.writerow([index, house])

if __name__ == "__main__":
	main()
