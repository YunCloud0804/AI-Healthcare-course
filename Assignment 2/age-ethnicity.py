import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from datetime import datetime
import matplotlib.cm as cm
import joypy

# Load datasets
patients = pd.read_csv('D:\homework\AI Healthcare\CSV\PATIENTS.csv')
adm = pd.read_csv('D:\homework\AI Healthcare\CSV\ADMISSIONS.csv')

# Merge the data using the 'SUBJECT_ID' column
merged_df = pd.merge(patients, adm, on='SUBJECT_ID', how='inner')

# Function to calculate age with handling for missing DOD and leap years
def calculate_age(DOB, DOD):
    def parse_date(date_str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    dob_date = parse_date(DOB)
    dod_date = parse_date(DOD) if not pd.isna(DOD) else datetime.now()

    return relativedelta(dod_date, dob_date).years if dob_date and dod_date else None

# Apply the calculate_age function
merged_df["age"] = merged_df.apply(lambda row: calculate_age(row["DOB"], row["DOD"]), axis=1)

# Filter the DataFrame to keep ages between 0 and 120, and non-null ages
filtered_df = merged_df[(merged_df['age'].between(0, 120)) & (~merged_df['age'].isna())]

# Select the top 6 ethnicities
top_ethnicities = merged_df['ETHNICITY'].value_counts().head(6).index
filtered_df = filtered_df[filtered_df['ETHNICITY'].isin(top_ethnicities)]

# Create a colormap for different ethnicities
color_map = cm.get_cmap('tab10', len(top_ethnicities))  # 'tab10' has 10 distinct colors

# Create the joy plot (ridgeline plot) with different colors for each ethnicity

joypy.joyplot(
    data=filtered_df,
    by='ETHNICITY',
    column='age',
    figsize=(16,9),
    overlap=1,
    grid=True,
    colormap=color_map  # Assign different colors to each ethnicity
)

# Add plot title and labels
plt.title('Age Distribution by Ethnicity (Top 6 Ethnicities, Ages 0-120)', fontsize=16)
plt.xlabel('Age', fontsize=12)

plt.tight_layout()
plt.savefig("age-ethnicity.svg", format="svg")
plt.show()
