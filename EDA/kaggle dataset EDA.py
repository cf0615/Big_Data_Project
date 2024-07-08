# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/Preprocessed/Kaggle_Sirio_preprocessed.csv'
data = pd.read_csv(file_path,)

# Function to describe data
def describe_data(data):
    print("Detailed Explanation of Data Types and Distribution of Data:\n")
    for column in data.columns:
        print(f"Attribute: {column}")
        print(f"Data Type: {data[column].dtype}")
        if data[column].dtype in ['float64', 'int64']:
            print(f"Summary Statistics:\n{data[column].describe()}\n")
        else:
            print(f"Value Counts:\n{data[column].value_counts()}\n")
        print("\n" + "-"*80 + "\n")


# Run the functions to display information
describe_data(data)


# Selecting relevant columns for visualization
selected_columns = ['GENDER', 'AGE_ABOVE65', 'DISEASE GROUPING 1', 'DISEASE GROUPING 2', 
                    'DISEASE GROUPING 3', 'DISEASE GROUPING 4', 'DISEASE GROUPING 5', 
                    'DISEASE GROUPING 6', 'ICU']

# Plotting the distribution of selected categorical/binary variables
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 16))
for i, col in enumerate(selected_columns):
    ax = axes[i // 3, i % 3]
    data[col].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()