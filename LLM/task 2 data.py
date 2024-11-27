import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# local dataset
diag = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\DIAGNOSES_ICD.csv')
pat = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\PATIENTS.csv')
adm = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\ADMISSIONS.csv')


merged_data = pd.merge(pat, adm, on="SUBJECT_ID", how="inner")
merged_data = pd.merge(merged_data, diag, on="SUBJECT_ID", how="inner")
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

merged_data["AGE"] = merged_data.apply(lambda row: calculate_age(row["DOB"], row["DOD"]), axis=1)
filtered_patients = merged_data[(merged_data['AGE'].between(0, 120)) & (~merged_data['AGE'].isna())]
#print(filtered_patients.info())
selected_codes = ['410', '411', '412', '413', '414', '426', '427', '428']
heart_patients = merged_data[merged_data["ICD9_CODE"].astype(str).str.startswith(tuple(selected_codes))]
columns_to_keep = ["SUBJECT_ID", "AGE", "GENDER", "ICD9_CODE", "HADM_ID_x", "HADM_ID_y"]
heart_patients_reduced = heart_patients[columns_to_keep]
print(heart_patients_reduced.info())
heart_patients_file = "D:\\homework\\AI Healthcare\\CSV\\heart_patients.csv"
heart_patients_reduced.to_csv(heart_patients_file, index=False)
pat = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\heart_patients.csv')
valid_patients = pat[(pat['AGE'] >= 0) & (pat['AGE'] <= 120)]
def truncate_icd9_code(code):
    try:
        return int(str(int(code))[:3])
    except (ValueError, TypeError):
        return None
pat['ICD9_CODE'] = pat['ICD9_CODE'].apply(truncate_icd9_code)
pat = pat.dropna(subset=['ICD9_CODE'])
print(pat.head())
updated_patients_file = "D:\\homework\\AI Healthcare\\CSV\\heart_patients_4xx.csv"
pat.to_csv(updated_patients_file, index=False)
pat = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\heart_patients_4xx.csv')
icd9_counts = pat['ICD9_CODE'].value_counts().reset_index()
icd9_counts.columns = ['ICD9_CODE', 'COUNT']
#print(icd9_counts)
patients_data = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\heart_patients_4xx.csv')
heart_rate_itemid = 212
chartevents_file = 'D:\\homework\\AI Healthcare\\CSV\\CHARTEVENTS.csv'
heart_rate_data = pd.DataFrame()
# use chunks to avoid memory problem
chunk_size = 1000000
for chunk in pd.read_csv(chartevents_file, chunksize=chunk_size, low_memory=False):
    filtered_chunk = chunk[
        (chunk['SUBJECT_ID'].isin(patients_data['SUBJECT_ID'])) &
        (chunk['ITEMID'] == heart_rate_itemid)
    ]
    heart_rate_data = pd.concat([heart_rate_data, filtered_chunk], ignore_index=True)
output_file = 'D:\\homework\\AI Healthcare\\LLM\\heart_rate_data.csv'
heart_rate_data.to_csv(output_file, index=False)
pat = pd.read_csv('D:\\homework\\AI Healthcare\\LLM\\heart_rate_data.csv')
columns_to_keep = ['SUBJECT_ID', 'ITEMID', 'VALUE']
heart_rate_cleaned = pat[columns_to_keep]
output_file = 'D:\\homework\\AI Healthcare\\LLM\\heart_rate.csv'
heart_rate_cleaned.to_csv(output_file, index=False)
pat = pd.read_csv('D:\\homework\\AI Healthcare\\LLM\\heart_rate.csv')
unique_values = pat['VALUE'].unique()
num_unique_values = len(unique_values)
'''
print(f"Number of unique types in the VALUE column: {num_unique_values}")
print("Unique types of values:")
print(unique_values)
value_counts = pat['VALUE'].value_counts()
print(f"Number of unique types in the VALUE column: {len(value_counts)}")
print("Counts for each unique type:")
print(value_counts)
'''


# get mapping data
value_mapping = {
    'Normal Sinus': 1, 'AV Paced': 2, 'Vent. Tachy': 3, 'Atrial Fib': 4,
    'Sinus Tachy': 5, 'V Paced': 6, 'Sinus Brady': 7, 'Supravent Tachy': 8,
    'Ventricular Fib': 9, 'A Paced': 10, '1st Deg AV Block': 11, '2nd AVB/Mobitz I': 12,
    'Atrial Flutter': 13, 'Sinus Arrhythmia': 14, None: 15, 'Other/Remarks': 16,
    'MultFocalAtrTach': 17, 'Junctional': 18, 'Asystole': 19,
    'Idioventricular': 20, 'Comp Heart Block': 21, 'Wand.Atrial Pace': 22,
    'Parox Atr Tachy': 23, 'Paced': 24, '2nd AVB Mobitz 2': 25, 'Zoll Paced': 26
}
pat['VALUE_MAPPED'] = pat['VALUE'].map(value_mapping)
updated_file = 'D:\\homework\\AI Healthcare\\LLM\\heart_rate_mapped.csv'
pat.to_csv(updated_file, index=False)

# get the note
pat = pd.read_csv('D:\\homework\\AI Healthcare\\LLM\\heart_rate_mapped.csv')
noteevents_file = 'D:\\homework\\AI Healthcare\\CSV\\NOTEEVENTS.csv'
heart_disease_subject_ids = pat['SUBJECT_ID'].unique()
filtered_notes = pd.DataFrame()

# use chunks to avoid memory problem
chunk_size = 1000000
columns_to_keep_from_notes = ['SUBJECT_ID', 'TEXT']

for chunk in pd.read_csv(noteevents_file, usecols=columns_to_keep_from_notes, chunksize=chunk_size, low_memory=False):
    # Filter rows for the relevant SUBJECT_IDs
    filtered_chunk = chunk[chunk['SUBJECT_ID'].isin(heart_disease_subject_ids)]
    # Append filtered rows
    filtered_notes = pd.concat([filtered_notes, filtered_chunk], ignore_index=True)
output_file = 'D:\\homework\\AI Healthcare\\LLM\\heart_rate_notes.csv'
filtered_notes.to_csv(output_file, index=False)


# for synthetic data

import random

num_records = 2000
# define random symptoms
symptoms_by_category = {
    1: "No significant abnormalities detected.",
    2: "Mild fatigue and occasional shortness of breath.",
    3: "Rapid heartbeat and dizziness.",
    4: "Irregular heartbeat with chest discomfort.",
    5: "Elevated pulse during mild physical activity.",
    6: "History of arrhythmias managed with medication.",
    7: "Slowed heart rate and light-headedness.",
    8: "Episodes of tachycardia and anxiety.",
    9: "Ventricular fibrillation detected, requiring monitoring.",
    10: "Atrial pacing with occasional palpitations.",
    11: "First-degree AV block with no symptoms.",
    12: "Second-degree AV block with fatigue and fainting.",
    13: "Atrial flutter causing shortness of breath.",
    14: "Sinus arrhythmia noted during physical exam.",
    15: "No definitive cardiac symptoms reported.",
    16: "Remarks noted, including atypical arrhythmias.",
    17: "Multifocal atrial tachycardia with dizziness.",
    18: "Junctional rhythm causing moderate fatigue.",
    19: "Asystole requiring immediate medical intervention.",
    20: "Idioventricular rhythm noted with mild symptoms.",
    21: "Complete heart block requiring pacemaker.",
    22: "Wandering atrial pacemaker detected.",
    23: "Paroxysmal atrial tachycardia episodes.",
    24: "Paced rhythm observed post-surgery.",
    25: "Second-degree AV block Mobitz II with syncope.",
}
# for synthetic data
synthetic_data = {
    "SUBJECT_ID": [random.randint(1, 100000) for _ in range(num_records)],  # Random subject IDs
    "ITEMID": [211] * num_records,  # Constant ITEMID for all records
    "VALUE": [random.randint(1, 25) for _ in range(num_records)],  # Random values between 1 and 25
    "TEXT": [
        f"Patient {random.randint(1, 100000)} exhibits symptoms: {symptoms_by_category[random_value]}"
        for random_value in [random.randint(1, 25) for _ in range(num_records)]
    ],  # Random detailed symptoms for each record
    "RISK_CLASS": [random.randint(1, 4) for _ in range(num_records)],  # Random classification 1-4
}

synthetic_df = pd.DataFrame(synthetic_data)
output_file = 'D:\\homework\\AI Healthcare\\LLM\\synthetic_data.csv'
synthetic_df.to_csv(output_file, index=False)


