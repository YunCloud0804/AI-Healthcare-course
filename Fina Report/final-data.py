import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


pat = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\PATIENTS.csv')
adm = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\ADMISSIONS.csv')
diag = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\DIAGNOSES_ICD.csv')

pat_adm = pd.merge(adm, pat, on='SUBJECT_ID', how='inner')
merged_data = pd.merge(pat_adm, diag, on='SUBJECT_ID', how='inner')
filtered_data = merged_data[merged_data['ICD9_CODE'].astype(str).str.startswith('584')].copy()

def calculate_age(DOB, DOD):
    def parse_date(date_str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
            except TypeError:
                return None
        return None

    dob_date = parse_date(DOB)
    dod_date = parse_date(DOD) if not pd.isna(DOD) else datetime.now()

    return relativedelta(dod_date, dob_date).years if dob_date and dod_date else None

filtered_data["AGE"] = filtered_data.apply(lambda row: calculate_age(row["DOB"], row["DOD"]), axis=1)
filtered_data = filtered_data[(filtered_data["AGE"] >= 0) & (filtered_data["AGE"] <= 120)]
filtered_data_unique = filtered_data.sort_values(by="ADMITTIME").groupby("SUBJECT_ID").first().reset_index()
filtered_data_final = filtered_data_unique[["SUBJECT_ID", "AGE", "GENDER"]]

# save file of step 1
filtered_data_final.to_csv('akf_patients.csv', index=False)



pat = pd.read_csv('akf_patients.csv')

merged_notes = []

# use chunk to avoid memory issue, merge with noteevents
chunk_size = 100000
noteevents_path = 'D:\\homework\\AI Healthcare\\CSV\\NOTEEVENTS.csv'

for chunk in pd.read_csv(noteevents_path, chunksize=chunk_size, low_memory=False):
    chunk = chunk[["SUBJECT_ID", "TEXT"]]
    merged_chunk = pd.merge(chunk, pat, on='SUBJECT_ID', how='inner')
    merged_notes.append(merged_chunk)

merged_notes_df = pd.concat(merged_notes, ignore_index=True)
merged_notes_combined = (
    merged_notes_df.groupby(["SUBJECT_ID", "AGE", "GENDER"])["TEXT"]
    .apply(lambda texts: " ".join(texts.dropna()))
    .reset_index()
)
# save file of step 2
merged_notes_combined.to_csv('akf_patients_notes.csv', index=False)




# merge with prescriptions
pat = pd.read_csv('akf_patients_notes.csv')
pre = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\PRESCRIPTIONS.csv', low_memory=False)

merged_data = pd.merge(pat, pre, on='SUBJECT_ID', how='inner')

filtered_data = merged_data[["SUBJECT_ID", "AGE", "GENDER", "TEXT", "DRUG"]]
grouped_data = (
    filtered_data.groupby(["SUBJECT_ID", "AGE", "GENDER", "TEXT"])["DRUG"]
    .apply(lambda drugs: drugs.dropna().unique().tolist())  # combine unique drug names into a list
    .reset_index()
)

# save file of step 3
grouped_data.to_csv('akf_patients_note_pre.csv', index=False)














