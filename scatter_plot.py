import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('ressources/dataset_train.csv')

fig, ax = plt.subplots()
colors = np.random.rand(1600)

ax.scatter(df["Astronomy"], df["Defense Against the Dark Arts"], alpha=0.3, c=colors)
ax.set_xlabel('Astronomy')
ax.set_ylabel('Defense Against The Dark Arts')
plt.show()
