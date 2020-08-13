import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import subprocess, os, platform

df = pd.read_csv('ressources/dataset_train.csv')

bar_color = {"Ravenclaw": "#2980b9",
            "Slytherin": "#27ae60",
            "Gryffindor": "#c0392b",
            "Hufflepuff": "#f1c40f"}

sns.pairplot(df, hue="Hogwarts House", markers = ".")
plt.tight_layout()
plt.savefig("pair_plot.png", format='png')

if platform.system() == 'Darwin':       # macOS
    subprocess.call(('open', './pair_plot.png'))
elif platform.system() == 'Windows':    # Windows
    os.startfile('./pair_plot.png')
else:                                   # linux variants
    subprocess.call(('xdg-open', './pair_plot.png'))
