import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('ressources/dataset_train.csv')

bar_color = {"Ravenclaw": "#2980b9",
            "Slytherin": "#27ae60",
            "Gryffindor": "#c0392b",
            "Hufflepuff": "#f1c40f"}

fig, ax = plt.subplots()
colors = np.random.rand(1600)

sns.pairplot(df, hue="Hogwarts House", palette=bar_color)
plt.savefig("pair_plot.png", dpi=300, format='png', bbox_inches='tight')
