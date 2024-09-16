import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data

# Merge icustays, admissions, and patients on SUBJECT_ID
merged_data_icu = pd.merge(icu, adm, on='SUBJECT_ID', how='inner')

# Ensure that the 'INTIME' column is in datetime format
merged_data_icu['INTIME'] = pd.to_datetime(merged_data_icu['INTIME'], errors='coerce')

# Extract year from 'INTIME' and create 'intime_year' column
merged_data_icu['intime_year'] = merged_data_icu['INTIME'].dt.year

# Filter the data to exclude years beyond 2200
merged_data_icu = merged_data_icu[merged_data_icu['intime_year'] <= 2200]

# Create a new column for 10-year intervals
merged_data_icu['year_interval'] = (merged_data_icu['intime_year'] // 10) * 10

# Find the top 6 ethnicities based on the number of occurrences
top_6_ethnicities = merged_data_icu['ETHNICITY'].value_counts().nlargest(6).index

# Filter the data for these top 6 ethnicities
filtered_data = merged_data_icu[merged_data_icu['ETHNICITY'].isin(top_6_ethnicities)]

# Group by ethnicity and 20-year interval, then calculate the mean LOS for each group
grouped_data = filtered_data.groupby(['year_interval', 'ETHNICITY'])['LOS'].mean().reset_index()

# Create a pivot table for the heatmap
pivot_data = grouped_data.pivot("ETHNICITY", "year_interval", "LOS")

# Set up the figure
plt.figure(figsize=(16, 9))
sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", linewidths=0.5)

# Set plot title and labels
plt.title("Heatmap of Mean LOS of All Careunit by Ethnicity and 10-Year Intervals (in days)")
plt.xlabel("10-Year Interval")
plt.ylabel("Ethnicity")

# Show the plot
plt.tight_layout()
plt.savefig("los-ethnicity.svg", format="svg")
plt.show()
