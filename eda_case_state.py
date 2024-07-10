# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:38:14 2024

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/User/.spyder-py3/caseState_preprocessed.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Summary statistics
summary_stats = data.describe()
print(summary_stats)

# Plotting the distribution of new cases
plt.figure(figsize=(10, 6))
plt.hist(data['cases_new'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of New Cases')
plt.xlabel('Number of New Cases')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plotting box plot for new cases to understand spread and outliers
plt.figure(figsize=(10, 6))
plt.boxplot(data['cases_new'], vert=False)
plt.title('Box Plot of New Cases')
plt.xlabel('Number of New Cases')
plt.grid(True)
plt.show()

# Bar chart for age group distribution
age_groups = ['cases_0_4', 'cases_5_11', 'cases_12_17', 'cases_18_29', 'cases_30_39', 'cases_40_49', 'cases_50_59', 'cases_60_69', 'cases_70_79', 'cases_80']
age_group_counts = data[age_groups].sum()

plt.figure(figsize=(12, 8))
age_group_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Distribution of Cases by Age Groups')
plt.xlabel('Age Groups')
plt.ylabel('Number of Cases')
plt.grid(True)
plt.show()

# Bar chart for vaccination status
vax_status = ['cases_unvax', 'cases_pvax', 'cases_fvax', 'cases_boost']
vax_counts = data[vax_status].sum()

plt.figure(figsize=(10, 6))
vax_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
plt.title('Distribution of Cases by Vaccination Status')
plt.xlabel('Vaccination Status')
plt.ylabel('Number of Cases')
plt.grid(True)
plt.show()

# Convert date to datetime format for better handling
data['date'] = pd.to_datetime(data['date'])

# Grouping data by date and state to get the time series data
grouped_data = data.groupby(['date', 'state']).sum().reset_index()

# Plotting new cases over time by state with color-coded legend
plt.figure(figsize=(15, 10))
for state in grouped_data['state'].unique():
    state_data = grouped_data[grouped_data['state'] == state]
    plt.plot(state_data['date'], state_data['cases_new'], label=state)

plt.title('New Cases Over Time by State')
plt.xlabel('Date')
plt.ylabel('Number of New Cases')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.grid(True)
plt.show()

# Plotting recovered cases over time by state with color-coded legend
plt.figure(figsize=(15, 10))
for state in grouped_data['state'].unique():
    state_data = grouped_data[grouped_data['state'] == state]
    plt.plot(state_data['date'], state_data['cases_recovered'], label=state)

plt.title('Recovered Cases Over Time by State')
plt.xlabel('Date')
plt.ylabel('Number of Recovered Cases')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.grid(True)
plt.show()

# Plotting active cases over time by state with color-coded legend
plt.figure(figsize=(15, 10))
for state in grouped_data['state'].unique():
    state_data = grouped_data[grouped_data['state'] == state]
    plt.plot(state_data['date'], state_data['cases_active'], label=state)

plt.title('Active Cases Over Time by State')
plt.xlabel('Date')
plt.ylabel('Number of Active Cases')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.grid(True)
plt.show()





