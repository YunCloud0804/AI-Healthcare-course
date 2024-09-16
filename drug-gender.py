import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter

# If the file is in the root project folder
patients = pd.read_csv('D:\homework\AI Healthcare\CSV\PATIENTS.csv')
pre = pd.read_csv('D:\homework\AI Healthcare\CSV\PRESCRIPTIONS.csv', low_memory=False)

df = patients

pre['DRUG'] = pre['DRUG'].str.strip().dropna()

# Convert the DataFrame to a dictionary
drug_dict = pre.groupby('DRUG').agg(list).to_dict(orient='index')

# Merge diagnosis_icd with patients admission on subject_id
merged_data_diagnosis = pd.merge(pre, patients, on = 'SUBJECT_ID', how = 'inner')

# get the top diagnosis codes for each gender
top_diagnosis_codes_male = merged_data_diagnosis[merged_data_diagnosis['GENDER'] == 'M']['DRUG'].value_counts().head(10)
top_diagnosis_codes_female = merged_data_diagnosis[merged_data_diagnosis['GENDER'] == 'F']['DRUG'].value_counts().head(10)

# Combine the male and female data into separate sets and sort them independently
male_data = pd.DataFrame({
    'drug': top_diagnosis_codes_male.index,
    'male_count': top_diagnosis_codes_male.values
}).sort_values(by='male_count', ascending=True)

female_data = pd.DataFrame({
    'drug': top_diagnosis_codes_female.index,
    'female_count': top_diagnosis_codes_female.values
}).sort_values(by='female_count', ascending=True)

# Normalize values for color mapping
norm_male = Normalize(vmin=0, vmax=male_data['male_count'].max())
norm_female = Normalize(vmin=0, vmax=female_data['female_count'].max())

# Define custom colormaps: starting from light blue for males and light red for females
light_blue = LinearSegmentedColormap.from_list("light_blue", ["#add8e6", "blue"])  # Light blue to dark blue
light_red = LinearSegmentedColormap.from_list("light_red", ["#ffcccc", "red"])    # Light red to dark red

# Create the butterfly plot
fig, ax_left = plt.subplots(figsize=(16, 9))

# Create a secondary axis for the right side
ax_right = ax_left.twinx()

# Function to plot gradient bars
def plot_gradient_barh(ax, drug, value, is_male=True):
    direction = -1 if is_male else 1
    cmap = light_blue if is_male else light_red
    norm = norm_male if is_male else norm_female
    norm_value = norm(value)

    # Create gradient as a series of small rectangles
    for i in range(100):  # 100 steps for smooth gradient
        color = cmap(norm_value * i / 100)  # Calculate color at each step
        ax.barh(drug, direction * value / 100, left=direction * value * i / 100, color=color, edgecolor=color, height=0.8)

# Plot male values as negative (left side) with light blue gradient on ax_left
for i, row in male_data.iterrows():
    plot_gradient_barh(ax_left, row['drug'], row['male_count'], is_male=True)

# Plot female values as positive (right side) with light red gradient on ax_right
for i, row in female_data.iterrows():
    plot_gradient_barh(ax_right, row['drug'], row['female_count'], is_male=False)

# Set y-axis labels and ticks for both sides
ax_left.set_yticks(male_data['drug'])
ax_left.set_yticklabels(male_data['drug'])
ax_right.set_yticks(female_data['drug'])
ax_right.set_yticklabels(female_data['drug'])

# Set x-axis ticks and modify x-axis ticks to display positive values on both sides
x_ticks = ax_left.get_xticks()
ax_left.xaxis.set_major_locator(FixedLocator(x_ticks))
ax_left.xaxis.set_major_formatter(FixedFormatter([str(abs(int(x))) for x in x_ticks]))

# Set labels
ax_left.set_xlabel('Number of Patients')
ax_left.set_ylabel('Drug Names (Male Side)')
ax_right.set_ylabel('Drug Names (Female Side)')
ax_left.set_title('Top 10 Drugs by Gender (Male and Female Sorted Independently)')

# Show the plot
plt.tight_layout()
plt.savefig("drug-gender.svg", format="svg")

plt.show()

