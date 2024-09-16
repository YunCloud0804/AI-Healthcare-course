
import pandas as pd
import matplotlib.pyplot as plt

# If the file is in the root project folder
adm = pd.read_csv('D:\homework\AI Healthcare\CSV\ADMISSIONS.csv')

pre = pd.read_csv('D:\homework\AI Healthcare\CSV\PRESCRIPTIONS.csv', low_memory=False)

pre['DRUG'] = pre['DRUG'].str.strip().dropna()

# Convert the DataFrame to a dictionary
drug_dict = pre.groupby('DRUG').agg(list).to_dict(orient='index')
ethn_dict = adm.groupby('ETHNICITY').agg(list).to_dict(orient='index')

# Merge diagnosis_icd with patients admission on subject_id
merged_data_diagnosis = pd.merge(pre, adm, on = 'SUBJECT_ID', how = 'inner')

ethnicity_drug_counts = merged_data_diagnosis.groupby(['ETHNICITY', 'DRUG']).size().reset_index(name='count')

# Step 2: Find the top 6 ethnicities based on the total number of occurrences
top_10_ethnicities = ethnicity_drug_counts.groupby('ETHNICITY')['count'].sum().nlargest(6).index

# Step 3: Filter the data to only include these top 6 ethnicities
top_ethnicity_drug_data = ethnicity_drug_counts[ethnicity_drug_counts['ETHNICITY'].isin(top_10_ethnicities)]

# Step 4: Find the top 10 most used drugs within these top 5 ethnicities
top_10_drugs = top_ethnicity_drug_data.groupby('DRUG')['count'].sum().nlargest(10).reset_index()

# Creat two graph for each one

# Step 1: Filter the top_ethnicity_drug_data to include only the top 10 drugs
top_drugs_filtered = top_ethnicity_drug_data[top_ethnicity_drug_data['DRUG'].isin(top_10_drugs['DRUG'])].copy()

# Step 2: Split the data into two groups: one for 'White' and 'Black', and one for the other 4 ethnicities
white_black_data = top_drugs_filtered[top_drugs_filtered['ETHNICITY'].isin(['WHITE', 'BLACK/AFRICAN AMERICAN'])].copy()
other_ethnicities_data = top_drugs_filtered[~top_drugs_filtered['ETHNICITY'].isin(['WHITE', 'BLACK/AFRICAN AMERICAN'])].copy()

# Step 3: Pivot the data for 'White' and 'Black' group with drugs on the X-axis
white_black_stacked_data = white_black_data.pivot(index='DRUG', columns='ETHNICITY', values='count').fillna(0)

# Step 4: For the other 4 ethnicities, sort ethnicities by total drug usage count (least to most)
ethnicity_totals = other_ethnicities_data.groupby('ETHNICITY')['count'].sum().sort_values()
sorted_ethnicities = ethnicity_totals.index.tolist()

# Step 5: Pivot the data for the remaining 4 ethnicities group and keep ethnicities sorted
other_stacked_data = other_ethnicities_data.pivot(index='DRUG', columns='ETHNICITY', values='count').fillna(0)

# Reorder columns (ethnicities) based on sorted order
other_stacked_data = other_stacked_data[sorted_ethnicities]

# Step 6: Plot for 'White' and 'Black' group with drugs on the X-axis
plt.figure(figsize=(16, 9))
white_black_stacked_data.plot(kind='bar', stacked=True)
plt.ylabel('Count of Drug Usage')
plt.xlabel('Drugs')
plt.title('Top 10 Drugs Usage by Top 2 Ethnicities')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("drug-ethn-2.svg", format="svg")
plt.show()

# Step 7: Plot for the other 4 ethnicities group with drugs on the X-axis and smaller legend
plt.figure(figsize=(16, 9))
other_stacked_data.plot(kind='bar', stacked=True)
plt.ylabel('Count of Drug Usage')
plt.xlabel('Drugs')
plt.title('Top 10 Drugs Usage by Other Ethnicities')
plt.xticks(rotation=45, ha='right')

# Adjust the legend with a smaller fontsize
plt.legend(title='Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.tight_layout()
plt.savefig("drug-ethn-4.svg", format="svg")
plt.show()

# Create graph usage per person

# Step 1: Find the total number of occurrences (or patients) per ethnicity
ethnicity_counts = merged_data_diagnosis.groupby('ETHNICITY')['SUBJECT_ID'].nunique().reset_index(name='num_people')

# Step 2: Merge ethnicity counts with the drug usage data to normalize the drug usage per person
ethnicity_drug_per_person = pd.merge(ethnicity_drug_counts, ethnicity_counts, on='ETHNICITY')

# Step 3: Calculate the drug usage per person by dividing the 'count' by 'num_people'
ethnicity_drug_per_person['usage_per_person'] = ethnicity_drug_per_person['count'] / ethnicity_drug_per_person['num_people']

# Step 4: Filter to only include the top 10 drugs and the top 6 ethnicities
top_drugs_filtered_per_person = ethnicity_drug_per_person[
    (ethnicity_drug_per_person['DRUG'].isin(top_10_drugs['DRUG'])) &
    (ethnicity_drug_per_person['ETHNICITY'].isin(top_10_ethnicities))
]

# Step 5: Pivot the data for plotting
drug_usage_per_person_data = top_drugs_filtered_per_person.pivot(index='DRUG', columns='ETHNICITY', values='usage_per_person').fillna(0)

# Step 6: Plot the graph for top 10 drugs usage by top 6 ethnicities per person
plt.figure(figsize=(16, 9))
drug_usage_per_person_data.plot(kind='bar', stacked=True)
plt.ylabel('Drug Usage Per Person')
plt.xlabel('Drugs')
plt.title('Top 10 Drugs Usage by Top 6 Ethnicities Per Person')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("drug-ethn-per.svg", format="svg")
plt.show()
