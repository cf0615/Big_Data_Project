# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:10:13 2024

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/User/.spyder-py3/cases_malaysia_preprocessed.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
data_info = data.info()
data_head = data.head()
data_tail = data.tail()
data_nunique = data.nunique()

data_info, data_head, data_tail
data_nunique

# Summary statistics for numerical columns
summary_stats = data.describe()

# Plotting histograms for numerical columns to understand their distributions
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(15, 20))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(9, 4, i)
    plt.hist(data[column], bins=20, edgecolor='k')
    plt.title(column)
    plt.tight_layout()

plt.show()

summary_stats

# Converting 'date' column to datetime format for time series analysis
data['date'] = pd.to_datetime(data['date'])

# Time series plot for key columns
plt.figure(figsize=(15, 10))

# Plotting new cases over time
plt.subplot(3, 1, 1)
plt.plot(data['date'], data['cases_new'], label='New Cases')
plt.title('New COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()

# Plotting active cases over time
plt.subplot(3, 1, 2)
plt.plot(data['date'], data['cases_active'], label='Active Cases', color='orange')
plt.title('Active COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()

# Plotting recovered cases over time
plt.subplot(3, 1, 3)
plt.plot(data['date'], data['cases_recovered'], label='Recovered Cases', color='green')
plt.title('Recovered COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()

plt.tight_layout()
plt.show()

# Correlation matrix
correlation_matrix = data.corr()

# Plotting the correlation matrix
plt.figure(figsize=(15, 15))
plt.matshow(correlation_matrix, fignum=1, cmap='coolwarm')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.colorbar()
plt.title('Correlation Matrix', pad=90)
plt.show()

correlation_matrix


