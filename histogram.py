import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def print_histograms(axs):
    axs[0, 0].patch.set_edgecolor('#2ecc71')
    axs[0, 0].patch.set_linewidth('3')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()

def draw_histograms(howgwarts_lessons, axs):
    bar_color = ["#c0392b", "#f1c40f", "#2980b9", "#27ae60"]
    i = 0

    while i < len(howgwarts_lessons.columns):
        lesson_grades = howgwarts_lessons.iloc[:, i].dropna()
        for house_index, house in enumerate(hogwarts_houses):
            filter = df["Hogwarts House"] == house
            house_grades = lesson_grades[filter]
            if i < 5:
                axs[i, 0].hist(house_grades, 10, label=house, color=bar_color[house_index], alpha=0.7)
                axs[i, 0].set_title(howgwarts_lessons.columns[i])
            elif i >= 5 and i < 10:
                axis = i - 5
                axs[axis, 1].hist(house_grades, 10, label=house, color=bar_color[house_index], alpha=0.7)
                axs[axis, 1].set_title(howgwarts_lessons.columns[i])
            else:
                axis = i - 10
                axs[axis, 2].hist(house_grades, 10, label=house, color=bar_color[house_index], alpha=0.7)
                axs[axis, 2].set_title(howgwarts_lessons.columns[i])
        i+=1

# Import dataset
df = pd.read_csv('ressources/dataset_train.csv')
df = df.drop(columns=["Index"], axis="column")
hogwarts_houses = list(set(df["Hogwarts House"]))
hogwarts_houses.sort()

# Create plot
plt.rcParams["figure.figsize"] = [14,8]
fig, axs = plt.subplots(5, 3)
fig.delaxes(axs[4,2])
fig.delaxes(axs[3,2])
axs[4, 1].set_xlabel("Grade")
axs[4, 1].set_ylabel("Number of students")
plt.tight_layout()
plt.subplots_adjust(top=0.92)

howgwarts_lessons = df.select_dtypes('number')
howgwarts_lessons=((howgwarts_lessons-howgwarts_lessons.min())/(howgwarts_lessons.max()-howgwarts_lessons.min()))*100

draw_histograms(howgwarts_lessons, axs)
print_histograms(axs)
