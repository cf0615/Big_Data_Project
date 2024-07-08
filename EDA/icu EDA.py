import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/Preprocessed/icu_preprocessed.csv'
data = pd.read_csv(file_path)

# Detailed explanation on the data type for each attribute
print("Data Types for Each Attribute:\n")
print(data.dtypes)
print("\n")

# Summary statistics to show distribution of data
summary_stats = data.describe()
print("Summary Statistics for Numerical Attributes:\n")
print(summary_stats)
print("\n")

# Additional statistics for non-numerical attributes (if any)
non_numeric_summary = data.describe(include=['object'])
if not non_numeric_summary.empty:
    print("Summary Statistics for Non-Numerical Attributes:\n")
    print(non_numeric_summary)
    print("\n")

# Distribution of data for each numerical attribute
for column in data.select_dtypes(include=['int64', 'float64']).columns:
    print(f"Distribution of {column}:\n")
    print(data[column].value_counts())
    print("\n")
    
# Plot settings
plt.figure(figsize=(12, 8))

# Histogram of ICU Beds (Total)
plt.subplot(2, 2, 1)
plt.hist(data['beds_icu_total'], bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of ICU Beds (Total)')
plt.xlabel('ICU Beds (Total)')
plt.ylabel('Frequency')

# Histogram of Ventilators (Total)
plt.subplot(2, 2, 2)
plt.hist(data['vent'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Histogram of Ventilators (Total)')
plt.xlabel('Ventilators (Total)')
plt.ylabel('Frequency')

# Box Plot of ICU Beds for COVID-19 Patients
plt.subplot(2, 2, 3)
plt.boxplot(data['beds_icu_covid'], vert=False)
plt.title('Box Plot of ICU Beds for COVID-19 Patients')
plt.xlabel('ICU Beds for COVID-19 Patients')

# Box Plot of ICU Patients (COVID-19, PUI, Non-COVID-19)
plt.subplot(2, 2, 4)
plt.boxplot([data['icu_covid'], data['icu_pui'], data['icu_noncovid']], vert=False, labels=['ICU COVID-19', 'ICU PUI', 'ICU Non-COVID-19'])
plt.title('Box Plot of ICU Patients')
plt.xlabel('Number of Patients')

plt.tight_layout()
plt.show()
