
import numpy as np
import pandas as pd

red_wine_data = pd.read_csv("./winequality-red.csv", delimiter=';')
white_wine_data = pd.read_csv("./winequality-white.csv", delimiter=';')

"""
for wine_quality in range(0, 11):
    tmp_counter = 0
    for element in range(0, len(red_wine_data)):
        if red_wine_data.loc[element, 'quality'] == wine_quality:
            tmp_counter = tmp_counter + 1

    print("Number of elements with index {} is {}".format(wine_quality, tmp_counter))

print("RED WINE DONE")
"""

# good wine (8, 9)
selected_data = []
for row in range(0, len(white_wine_data)):
    if white_wine_data.loc[row, 'quality'] == 8 or white_wine_data.loc[row, 'quality'] == 9:
        selected_data.append(white_wine_data.loc[row])

selected_dataFrame = pd.DataFrame(data=selected_data)
selected_dataFrame = selected_dataFrame.drop(columns=['quality'])
np.save("./good_white_wine.npy", selected_dataFrame)

# bad wine (3-7)
selected_data_bad = []
for row in range(0, len(white_wine_data)):
    if white_wine_data.loc[row, 'quality'] == 3 or white_wine_data.loc[row, 'quality'] == 4 \
            or white_wine_data.loc[row, 'quality'] == 5 or white_wine_data.loc[row, 'quality'] == 6 \
            or white_wine_data.loc[row, 'quality'] == 7:
        selected_data_bad.append(white_wine_data.loc[row])


selected_dataFrame_bad = pd.DataFrame(data=selected_data_bad)
selected_dataFrame_bad = selected_dataFrame_bad.drop(columns=['quality'])
np.save("./bad_white_wine.npy", selected_dataFrame_bad)

"""
for wine_quality in range(0, 11):
    tmp_counter = 0
    for element in range(0, len(white_wine_data)):
        if white_wine_data.loc[element, 'quality'] == wine_quality:
            tmp_counter = tmp_counter + 1

    print("Number of elements with index {} is {}".format(wine_quality, tmp_counter))

print("WHITE WINE DONE")
"""
