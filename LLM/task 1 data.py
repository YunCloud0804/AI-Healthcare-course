import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta



adm = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\ADMISSIONS.csv')
pat = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\PATIENTS.csv')
diag = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\DIAGNOSES_ICD.csv')
patients_admissions = pd.merge(pat, adm, on="SUBJECT_ID", how="inner")
full_data = pd.merge(patients_admissions, diag, on="SUBJECT_ID", how="inner")
diabetes_patients = full_data[full_data['ICD9_CODE'].astype(str).str.startswith('250')]
diabetes_patients.to_csv('D:\\homework\\AI Healthcare\\CSV\\diabetes_patients.csv', index=False)
diabetes_pat = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\diabetes_patients.csv')

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
diabetes_pat["AGE"] = diabetes_pat.apply(lambda row: calculate_age(row["DOB"], row["DOD"]), axis=1)
filtered_diabetes_patients = diabetes_pat[(diabetes_pat['AGE'].between(0, 120)) & (~diabetes_pat['AGE'].isna())] # filter the age
filtered_diabetes_patients.to_csv('D:\\homework\\AI Healthcare\\CSV\\diabetes_pat_age.csv', index=False)
dia_pat = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\diabetes_pat_age.csv')
labevents = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\LABEVENTS.csv')
items = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\D_LABITEMS.csv')
labevents['ITEMID'] = labevents['ITEMID'].astype(int)
items['ITEMID'] = items['ITEMID'].astype(int)
# mapping itemid to label
itemid_to_label = items.set_index('ITEMID')['LABEL'].to_dict()
labevents['LABEL'] = labevents['ITEMID'].map(itemid_to_label) # add a column
#print(labevents.info())
labevents.to_csv('D:\\homework\\AI Healthcare\\CSV\\labevents_with_labels.csv', index=False)
dia_pat = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\diabetes_pat_age.csv')
lab = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\labevents_with_labels.csv')
# read part of the data
dia_pat = dia_pat[['SUBJECT_ID', 'ICD9_CODE', 'AGE']]
lab = lab[['SUBJECT_ID', 'ITEMID', 'VALUE', 'LABEL']]
dia_pat['SUBJECT_ID'] = dia_pat['SUBJECT_ID'].astype(int)
lab['SUBJECT_ID'] = lab['SUBJECT_ID'].astype(int)
# merge the data
merged_df = pd.merge(dia_pat, lab, on="SUBJECT_ID", how="inner")
filtered_df = merged_df[merged_df['ICD9_CODE'].astype(str).str.startswith('250')]
#print(filtered_df.info())
filtered_df.to_csv('D:\\homework\\AI Healthcare\\CSV\\diabetes_patients_with_labs.csv', index=False)
file_path = 'D:\\homework\\AI Healthcare\\CSV\\diabetes_patients_with_labs.csv'
output_path = 'D:\\homework\\AI Healthcare\\CSV\\blood_sugar_tests.csv'
blood_sugar_itemids = [50809, 226537, 227015]  # related itemid
filtered_chunks = []
# use chunks to avoid memory issue
chunk_size = 100000
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    filtered_chunk = chunk[chunk['ITEMID'].isin(blood_sugar_itemids)]
    filtered_chunks.append(filtered_chunk)
blood_sugar_tests = pd.concat(filtered_chunks, ignore_index=True)
blood_sugar_tests.to_csv(output_path, index=False)
#print(blood_sugar_tests.head())




# synthetic data
import random

num_samples = 1000
random.seed(42)
subject_ids = [random.randint(1, 100000) for _ in range(num_samples)]
blood_sugar_values = [random.randint(50, 350) for _ in range(num_samples)]  # Random blood sugar values
def classify_blood_sugar(value):
    if value < 140:
        return "Low Risk"
    elif 140 <= value <= 199:
        return "Risk"
    else:
        return "High Risk"
categories = [classify_blood_sugar(value) for value in blood_sugar_values]
synthetic_data = pd.DataFrame({
    "SUBJECT_ID": subject_ids,
    "RANDOM_BLOOD_SUGAR": blood_sugar_values,
    "CLASSIFICATION": categories
})
output_file = "synthetic_data_diabetes.csv"
synthetic_data.to_csv(output_file, index=False)
syn_data = pd.read_csv('D:\\homework\\AI Healthcare\\LLM\\synthetic_data_diabetes.csv')
print(syn_data.info())

















