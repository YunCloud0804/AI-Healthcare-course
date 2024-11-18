import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample


lab = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\LABEVENTS_reduced.csv', low_memory=False)
adm = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\ADMISSIONS.csv', low_memory=False)

data = pd.merge(lab, adm[['HADM_ID', 'ADMISSION_TYPE']], on='HADM_ID', how='inner')
label_mapping = {  # encode
    "EMERGENCY": 0,
    "ELECTIVE": 1,
    "NEWBORN": 2,
    "URGENT": 3
}
data['LABEL'] = data['ADMISSION_TYPE'].map(label_mapping)
data['VALUE'] = pd.to_numeric(data['VALUE'], errors='coerce')
data = data.dropna(subset=['VALUE', 'LABEL'])
data['LABEL'] = data['LABEL'].astype(int)
print("Original class distribution:")
print(data['LABEL'].value_counts())

majority_class = data[data['LABEL'] == 0]
minority_classes = data[data['LABEL'] != 0]

# downsample, take 25% of emergency
downsampled_majority = resample(majority_class, replace=False, n_samples=int(len(majority_class) * 0.25), random_state=42)
balanced_data = pd.concat([downsampled_majority, minority_classes])
print("Balanced class distribution:")
print(balanced_data['LABEL'].value_counts())

scaler = MinMaxScaler()
balanced_data['VALUE'] = scaler.fit_transform(balanced_data[['VALUE']]) # scaling
features = balanced_data[['ITEMID', 'VALUE']]
labels = balanced_data['LABEL']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))
