import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('ressources/dataset_train.csv')

fig, ax = plt.subplots()
colors = np.random.rand(1600)

ax.scatter(df["Astronomy"], df["Defense Against the Dark Arts"], alpha=0.3, c=colors)
plt.savefig("scatter_plot.png", dpi=300, format='png', bbox_inches='tight')
