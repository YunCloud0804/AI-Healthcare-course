import pandas as pd
import dask.dataframe as dd


lab = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\labevents.csv', low_memory=False)
adm = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\admissions.csv', low_memory=False)
pat = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\patients.csv', low_memory=False)
diag = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\diagnoses_icd.csv', low_memory=False)
item = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\d_labitems.csv', low_memory=False)
pre = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\prescriptions.csv', low_memory=False)

print("data loaded")
print(lab.info())
print(adm.info())
print(pat.info())
print(diag.info())
print(item.info())
print(pre.info())

akf_diag = diag[diag['icd_code'].str.startswith('584', na=False)]

akf_adm = akf_diag.merge(adm, on='hadm_id', how='inner', suffixes=('', '_adm'))
if 'subject_id_adm' in akf_adm.columns:
    akf_adm['subject_id'] = akf_adm['subject_id_adm']
    akf_adm.drop(columns=['subject_id_adm'], inplace=True)
akf_patients = akf_adm.merge(pat, on='subject_id', how='inner')

# select patients with 584
output_path = 'D:\\homework\\AI Healthcare\\CSV\\new\\akf_patient.csv'
akf_patients.to_csv(output_path, index=False)

aki_patients_file = 'D:\\homework\\AI Healthcare\\CSV\\new\\akf_patient.csv'
lab_file = 'D:\\homework\\AI Healthcare\\CSV\\new\\labevents.csv'
output_file = 'D:\\homework\\AI Healthcare\\CSV\\new\\lab_reduced.csv'

aki_patients = dd.read_csv(aki_patients_file)
aki_ids = aki_patients[['subject_id', 'hadm_id']]  # Use only necessary columns

lab = dd.read_csv(
    lab_file,
    usecols=['subject_id', 'hadm_id', 'itemid', 'value'],  # we only need this
    low_memory=False,
    dtype={
        'subject_id': 'int64',
        'hadm_id': 'float64',
        'itemid': 'int64',
        'value': 'object'
    }
)
filtered_char = lab.merge(aki_ids, on=['subject_id', 'hadm_id'], how='inner')
filtered_char.to_csv(output_file, single_file=True, index=False)

# save a pre_temp file with what we need
pre_file = 'D:\\homework\\AI Healthcare\\CSV\\new\\prescriptions.csv'
temp_file = 'D:\\homework\\AI Healthcare\\CSV\\new\\prescriptions_temp.csv'
pre_temp = pd.read_csv(pre_file, usecols=['subject_id', 'drug', 'form_rx'], low_memory=False)
pre_temp.to_csv(temp_file, index=False)

pre = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\prescriptions_temp.csv', low_memory=False)
pat = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\akf_patient.csv', low_memory=False)

pat_filtered = pat[['subject_id', 'hadm_id', 'icd_code']]
pre_pat= pre.merge(pat_filtered, on='subject_id', how='inner')
output_path = 'D:\\homework\\AI Healthcare\\CSV\\new\\pre_pat.csv'
pre_pat.to_csv(output_path, index=False)
lab = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\lab_reduced.csv', low_memory=False)
item = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\d_labitems.csv', low_memory=False)
lab_with_labels = lab.merge(item[['itemid', 'label']], on='itemid', how='left')
output_path = 'D:\\homework\\AI Healthcare\\CSV\\new\\lab_with_labels.csv'
lab_with_labels.to_csv(output_path, index=False)

# now the following data is ready to use in the model
lab = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\lab_with_labels.csv', low_memory=False)
pre = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\pre_pat.csv', low_memory=False)
item = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\d_labitems.csv', low_memory=False)







