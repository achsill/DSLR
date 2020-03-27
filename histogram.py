import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('ressources/dataset_train.csv')
print(df.head())
df = df.drop(columns=["Index"], axis=1)

def convert_set_to_list(set):
    result_list = []
    for elem in set:
        result_list.append(elem)

    result_list.sort()
    return result_list

# std_list = []
hogwarts_houses = set(df["Hogwarts House"])
# for house in hogwarts_houses:
#     filter = df["Hogwarts House"] == house
#     filtered_data = df[filter]
#     filtered_data = filtered_data.select_dtypes('number')
#     filtered_data=((filtered_data-filtered_data.min())/(filtered_data.max()-filtered_data.min()))*100
#     i = 0
#     header_length = len(filtered_data.columns)
#     house_std_list = []
#     while i < header_length:
#         house_std_list.append(filtered_data.iloc[:, i].std())
#         i += 1
#     std_list.append(house_std_list)
#
# cleared_df = pd.DataFrame(np.array(std_list),
#                    columns=df.select_dtypes('number').columns,
#                    index=set(df["Hogwarts House"]))

i = 0
column_index = 0
std_min = np.nan
# cleared_df_stds = cleared_df.std()
fig, axs = plt.subplots(5, 3)
fig.delaxes(axs[4,2])
fig.delaxes(axs[3,2])


# while i < len(cleared_df_stds):
#     if np.isnan(std_min) == True:
#         column_index = i
#         std_min = cleared_df_stds[i]
#     elif cleared_df_stds[i] < std_min:
#         std_min = cleared_df_stds[i]
#         column_index = i
#     i += 1

colors = ["#c0392b", "#f1c40f", "#2980b9", "#27ae60"]

i = 0
filtered_data = df.select_dtypes('number')
hogwarts_houses = convert_set_to_list(hogwarts_houses)

while i < len(filtered_data.columns):
    curr = filtered_data.iloc[:, i]
    curr=((curr-curr.min())/(curr.max()-curr.min()))*100
    curr = curr.dropna()
    for c_index, house in enumerate(hogwarts_houses):
        filter = df["Hogwarts House"] == house
        curr_tmp = curr[filter]
        if i < 5:
            n, bins, patches  = axs[i, 0].hist(curr_tmp, 10, label=house, color=colors[c_index], alpha=0.7)
            axs[i, 0].set_xlabel(filtered_data.columns[i])
        elif i >= 5 and i < 10:
            axis = i - 5
            n, bins, patches  = axs[axis, 1].hist(curr_tmp, 10, label=house, color=colors[c_index], alpha=0.7)
            axs[axis, 1].set_xlabel(filtered_data.columns[i])
        else:
            axis = i - 10
            n, bins, patches  = axs[axis, 2].hist(curr_tmp, 10, label=house, color=colors[c_index], alpha=0.7)
            axs[axis, 2].set_xlabel(filtered_data.columns[i])
    i+=1
#
figure = plt.gcf()
figure.set_size_inches(19, 18)
plt.legend(bbox_to_anchor=(1, 1))
plt.savefig("new.png", dpi=300, format='png', bbox_inches='tight')
