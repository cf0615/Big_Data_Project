# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:06:00 2024

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'C:/Users/User/.spyder-py3/deaths_state.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
data_info = data.info()
data_head = data.head()
data_tail = data.tail()
data_nunique = data.nunique()

data_info, data_head, data_tail
data_nunique

# List of numerical columns
numerical_columns = [
    'deaths_new', 'deaths_bid', 'deaths_new_dod', 'deaths_bid_dod',
    'deaths_unvax', 'deaths_pvax', 'deaths_fvax', 'deaths_boost', 'deaths_tat'
]

# Plot histograms for numerical columns
plt.figure(figsize=(15, 10))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    plt.hist(data[column], bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Plot the distribution of the 'state' column
plt.figure(figsize=(12, 6))
data['state'].value_counts().plot(kind='bar', color='skyblue', edgecolor='k')
plt.title('Distribution of Data by State')
plt.xlabel('State')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

# Calculate total deaths by state
total_deaths_by_state = data.groupby('state')['deaths_new'].sum()

# Plot the total deaths by state
plt.figure(figsize=(12, 6))
total_deaths_by_state.plot(kind='bar', color='salmon', edgecolor='k')
plt.title('Total Deaths by State')
plt.xlabel('State')
plt.ylabel('Total Deaths')
plt.xticks(rotation=90)
plt.show()

# Compute the correlation matrix
correlation_matrix = data[numerical_columns].corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Columns')
plt.show()

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])


# Calculate cumulative deaths by state over time
cumulative_deaths_by_state = data.groupby(['date', 'state'])['deaths_new'].sum().groupby(level=1).cumsum().reset_index()

# Plot the cumulative deaths by state over time
plt.figure(figsize=(15, 10))

# Plot each state's cumulative deaths over time
for state in cumulative_deaths_by_state['state'].unique():
    state_data = cumulative_deaths_by_state[cumulative_deaths_by_state['state'] == state]
    plt.plot(state_data['date'], state_data['deaths_new'], label=state)

plt.title('Cumulative Deaths by State Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Deaths')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='State')
plt.grid(True)
plt.tight_layout()
plt.show()
