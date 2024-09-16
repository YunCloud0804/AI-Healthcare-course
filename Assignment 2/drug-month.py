import pandas as pd
import matplotlib.pyplot as plt

# Load the PRESCRIPTIONS dataset

# Clean the DRUG column
pre['DRUG'] = pre['DRUG'].str.strip().dropna()

# Ensure STARTDATE is in datetime format
pre['STARTDATE'] = pd.to_datetime(pre['STARTDATE'], errors='coerce')

# Drop rows where STARTDATE could not be converted to datetime
pre = pre.dropna(subset=['STARTDATE'])

# Extract the month from STARTDATE
pre['MONTH'] = pre['STARTDATE'].dt.strftime('%b')  # Use %b to get short month names (Jan, Feb, etc.)

# Find the top 5 drugs overall (all-time)
top_drugs = pre['DRUG'].value_counts().nlargest(10).index

# Filter the data to keep only the top 5 drugs
top_drug_data = pre[pre['DRUG'].isin(top_drugs)]

# Group the data by month and drug, and count the number of prescriptions for each drug per month
monthly_usage = top_drug_data.groupby(['MONTH', 'DRUG']).size().unstack(fill_value=0)

# Ensure that months appear in order (Jan, Feb, ..., Dec)
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_usage = monthly_usage.reindex(month_order)

# Plot the results
fig, ax = plt.subplots(figsize=(16, 9))

# Plot the usage for the top 10 drugs across different months
for drug in monthly_usage.columns:
    ax.plot(monthly_usage.index, monthly_usage[drug], marker='o', label=f'{drug}')

ax.set_xlabel('Month')
ax.set_ylabel('Drug Usage (Number of Prescriptions)')
ax.legend(loc='upper left', title='Drugs')
plt.title('Top 10 Drug Usage Across Different Months')
plt.tight_layout()
plt.savefig("drug-month.svg", format="svg")
plt.show()
