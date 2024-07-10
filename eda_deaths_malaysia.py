# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 19:56:45 2024

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'C:/Users/User/.spyder-py3/deaths_malaysia_preprocessed.csv'
data = pd.read_csv(file_path)

# Generate summary statistics
summary_statistics = data.describe(include='all').T

# Display the summary statistics
print(summary_statistics)

# Display basic information about the dataset
data_info = data.info()
data_head = data.head()
data_tail = data.tail()
data_nunique = data.nunique()

data_info, data_head, data_tail
data_nunique

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])
plt.figure(figsize=(10, 6))
sns.histplot(data['deaths_new'], bins=30, kde=True)
plt.title('Distribution of Total New Deaths')
plt.xlabel('Total New Deaths')
plt.ylabel('Frequency')
plt.show()

# Plot total new deaths over time
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['deaths_new'], label='New Deaths')
plt.title('Total New Deaths Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Deaths')
plt.legend()
plt.grid(True)
plt.show()

# Plot deaths by vaccination status over time
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['deaths_unvax'], label='Unvaccinated', color='red')
plt.plot(data['date'], data['deaths_pvax'], label='Partially Vaccinated', color='blue')
plt.plot(data['date'], data['deaths_fvax'], label='Fully Vaccinated', color='green')
plt.plot(data['date'], data['deaths_boost'], label='Boosted', color='purple')
plt.title('Deaths by Vaccination Status Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Deaths')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the scatter plot for total new deaths vs vaccinated deaths
plt.figure(figsize=(10, 6))
plt.scatter(data['deaths_new'], data['deaths_pvax'] + data['deaths_fvax'] + data['deaths_boost'])
plt.title('Scatter Plot of Total New Deaths vs Vaccinated Deaths')
plt.xlabel('Total New Deaths')
plt.ylabel('Vaccinated Deaths')
plt.show()

# Compute correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, fignum=1)
plt.colorbar()
plt.title('Correlation Matrix', pad=20)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()

# Generate a scatter plot comparing 'deaths_new' and 'deaths_unvax'
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['deaths_new'], y=data['deaths_unvax'])
plt.title('Scatter Plot of Total New Deaths vs Unvaccinated Deaths')
plt.xlabel('Total New Deaths')
plt.ylabel('Unvaccinated Deaths')
plt.show()