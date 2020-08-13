import pandas as pd
import math
import numpy as np
import os.path
from os import path
import sys


def isNaN(num):
    return num != num

def length(list):
    count = 0
    for elem in list:
        if isNaN(elem) == 0:
            count += 1
    return round(count, 5)

def mean(list):
    result = 0
    for elem in list:
        if isNaN(elem) == 0:
            result = result + elem
    return round((result / length(list)), 6)

def std(list):
    result = 0
    mean_value = mean(list)
    for elem in list:
        if isNaN(elem) == 0:
            result = result + abs(elem - mean_value)**2
    lengthlol = length(list) - 1
    return round(math.sqrt(result / lengthlol), 6)

def min_value(list):
    result = None
    for elem in list:
        if isNaN(elem) == 0:
            if result == None or elem < result:
                result = elem
    return round(result, 6)

def max_value(list):
    result = None
    for elem in list:
        if isNaN(elem) == 0:
            if result == None or elem > result:
                result = elem
    return round(result, 6)

def compute_quartile(list, quartile):
    list_size = length(list)
    list = list.reset_index(drop=True)

    x = (quartile / 100) * (list_size - 1) + 1
    x_int = int(x)

    result = list.at[x_int - 1] + (x - x_int) * (list.at[x_int] - list.at[x_int - 1])
    return round(result, 6)

def find_longest_element_length(list, header_length):
    list_size = len(list)
    i = 0
    j = 0
    longest_element_length = 0
    format_string = ""
    while j < header_length:
        while i < list_size:
            if (len(str(list[i][j])) > longest_element_length):
                longest_element_length = len(str(list[i][j]))
            i += 1
        j += 1
        i = 0
        if format_string == "":
            format_string = format_string + "{:<" + str(longest_element_length + 2) + "}"
        else:
            format_string = format_string + "{:>" + str(longest_element_length + 2) + "}"
        longest_element_length = 0
    return format_string

def main():
    if len(sys.argv) < 2:
        print('This program needs a dataset')
        exit(0)
    if path.exists(sys.argv[1]) == False:
        print('This file doesn\'t exist')
        exit(0)
    df = pd.read_csv(sys.argv[1])
    df = df.select_dtypes('number')

    count_list = ["count"]
    mean_list = ["mean"]
    std_list = ["std"]
    min_list = ["min"]
    max_list = ["max"]
    first_quartile_list = ["25%"]
    second_quartile_list = ["50%"]
    third_quartile_list = ["75%"]
    result = []
    header = list(df.columns)
    header_length = len(header)
    i = 0

    while i < header_length:
        count_list.append(length(df.iloc[:, i]))
        mean_list.append(mean(df.iloc[:, i]))
        std_list.append(std(df.iloc[:, i]))
        min_list.append(min_value(df.iloc[:, i]))
        max_list.append(max_value(df.iloc[:, i]))
        first_quartile_list.append(compute_quartile(df.iloc[:, i].sort_values(), 25))
        second_quartile_list.append(compute_quartile(df.iloc[:, i].sort_values(), 50))
        third_quartile_list.append(compute_quartile(df.iloc[:, i].sort_values(), 75))
        i += 1

    header.insert(0, "")
    result.extend((header, count_list, mean_list, std_list, min_list, first_quartile_list, second_quartile_list, third_quartile_list, max_list))

    for row in result:
        print(find_longest_element_length(result, header_length + 1).format(*row))

if __name__ == "__main__":
    main()
