# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/Preprocessed/covidDataPreprocessed.csv'
data = pd.read_csv(file_path)

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

# Plotting the distribution of categorical/binary variables
categorical_columns = ['SEX', 'PNEUMONIA', 'DIABETES', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE',
                       'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU', 'RESULT']

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 16))
for i, col in enumerate(categorical_columns):
    ax = axes[i // 3, i % 3]
    data[col].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()

# Plotting the distribution of the continuous variable (AGE)
plt.figure(figsize=(10, 6))
data['AGE'].plot(kind='hist', bins=20, edgecolor='black')
plt.title('Distribution of AGE')
plt.xlabel('AGE')
plt.ylabel('Frequency')
plt.show()

