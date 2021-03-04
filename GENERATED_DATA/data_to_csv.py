
import numpy as np
import pandas as pd

csv_header = ["feature_01", "feature_02", "feature_03", "feature_04", "feature_05", "feature_06", "feature_07",
              "feature_08", "feature_09", "feature_10", "feature_11", "feature_12", "feature_13", "feature_14",
              "feature_15", "feature_16", "feature_17", "feature_18", "feature_19", "feature_20", "class"]

file_1 = np.load("./file1.npy")
file_2 = np.load("./file2.npy")
file_3 = np.load("./file3.npy")

# store to csv
pd.DataFrame(file_1).to_csv("./file1.csv", header=csv_header, index=False)
pd.DataFrame(file_2).to_csv("./file2.csv", header=csv_header, index=False)
pd.DataFrame(file_3).to_csv("./file3.csv", header=csv_header, index=False)
