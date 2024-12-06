import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
drug = pd.read_csv('top_20_prescriptions.csv')  # Update this path
drug_pre = pd.read_csv('top_20_predicted_drugs.csv')  # Update this path

# Ensure column names are correct
drug.columns = ['Drug', 'Count']
drug_pre.columns = ['Drug_pre', 'Count_pre']

# Sort both datasets by their counts in descending order
drug = drug.sort_values(by='Count', ascending=False).reset_index(drop=True)
drug_pre = drug_pre.sort_values(by='Count_pre', ascending=False).reset_index(drop=True)

# First horizontal bar graph: Actual data (drug)
plt.figure(figsize=(16, 9))
positions = np.arange(len(drug['Drug']))
plt.barh(positions, drug['Count'], color='blue', alpha=0.7, label='Actual Counts')
plt.yticks(positions, drug['Drug'])
plt.xlabel('Count')
plt.title('Horizontal Bar Graph: Actual Counts (Drug)')
plt.gca().invert_yaxis()  # Invert y-axis so largest is at the top
plt.tight_layout()
plt.savefig('drug_actual.jpg')
plt.show()

# Second horizontal bar graph: Predicted data (drug_pre)
plt.figure(figsize=(16, 9))
positions = np.arange(len(drug_pre['Drug_pre']))
plt.barh(positions, drug_pre['Count_pre'], color='purple', alpha=0.7, label='Predicted Counts')
plt.yticks(positions, drug_pre['Drug_pre'])
plt.xlabel('Count')
plt.title('Horizontal Bar Graph: Predicted Counts (Drug_pre)')
plt.gca().invert_yaxis()  # Invert y-axis so largest is at the top
plt.tight_layout()
plt.savefig('drug_predicted.jpg')
plt.show()
