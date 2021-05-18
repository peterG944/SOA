
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

bank_marketing_data = pd.read_csv("./bank-additional.csv", delimiter=';')

label_encoder = LabelEncoder()
columns_to_encode = ['loan', 'contact', 'month', 'day_of_week', 'poutcome', 'housing', 'default', 'marital', 'education']
for column in columns_to_encode:
    bank_marketing_data[column] = label_encoder.fit_transform(bank_marketing_data[column])


# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(bank_marketing_data[['job']]).toarray())
bank_marketing_data = bank_marketing_data.join(enc_df)

bank_marketing_data = bank_marketing_data.drop(columns=['job'])

# filter data
selected_data_correct = []
selected_data_incorrect = []
for row in range(0, len(bank_marketing_data)):
    if bank_marketing_data.loc[row, 'y'] == 'no':
        selected_data_correct.append(bank_marketing_data.loc[row])

    if bank_marketing_data.loc[row, 'y'] == 'yes':
        selected_data_incorrect.append(bank_marketing_data.loc[row])


selected_dataFrame_correct = pd.DataFrame(data=selected_data_correct)
selected_dataFrame_correct = selected_dataFrame_correct.drop(columns=['y'])
np.save("./bank_marketing_majority.npy", selected_dataFrame_correct)

selected_dataFrame_incorrect = pd.DataFrame(data=selected_data_incorrect)
selected_dataFrame_incorrect = selected_dataFrame_incorrect.drop(columns=['y'])
np.save("./bank_marketing_minority.npy", selected_dataFrame_incorrect)

