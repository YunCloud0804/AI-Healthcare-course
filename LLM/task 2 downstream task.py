import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import json

note_file = 'D:\\homework\\AI Healthcare\\LLM\\heart_rate_notes.csv'
hr_file = 'D:\\homework\\AI Healthcare\\LLM\\heart_rate_mapped.csv'
last_layer_file = 'D:\\homework\\AI Healthcare\\LLM\\gpt2_last_layer_embeddings.json'

note = pd.read_csv(note_file)
note_sampled = note.sample(frac=0.5, random_state=42).reset_index(drop=True)
hr = pd.read_csv(hr_file)
filtered_subject_ids = note_sampled["SUBJECT_ID"].unique()
hr_filtered = hr[hr["SUBJECT_ID"].isin(filtered_subject_ids)].reset_index(drop=True)

# load gpt2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# get features
print("Extracting note embeddings in batches...")
batch_size = 64
note_embeddings = []
for i in tqdm(range(0, len(note_sampled), batch_size), desc="Processing batches"):
    batch_texts = note_sampled["TEXT"][i:i+batch_size].tolist()
    inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=256, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        note_embeddings.extend(batch_embeddings)

note_sampled["EMBEDDING"] = list(note_embeddings)
np.save("note_embeddings.npy", np.array(note_embeddings))
note_sampled.to_pickle("note_sampled_with_embeddings.pkl")

# load the data and result
note_sampled_file = "note_sampled_with_embeddings.pkl"
last_layer_file = "D:\\homework\\AI Healthcare\\LLM\\gpt2_last_layer_embeddings.json"
hr_file = 'D:\\homework\\AI Healthcare\\LLM\\heart_rate_mapped.csv'

note_sampled = pd.read_pickle(note_sampled_file)
hr_filtered = pd.read_csv(hr_file)

# mapping
risk_mapping = {
    1: ("Normal Sinus", "Low Risk"),
    2: ("AV Paced", "Mild Symptoms"),
    3: ("Vent. Tachy", "Severe Symptoms"),
    4: ("Atrial Fib", "Severe Symptoms"),
    5: ("Sinus Tachy", "Mild Symptoms"),
    6: ("V Paced", "Mild Symptoms"),
    7: ("Sinus Brady", "Mild Symptoms"),
    8: ("Supravent Tachy", "Severe Symptoms"),
    9: ("Ventricular Fib", "Critical Condition"),
    10: ("A Paced", "Mild Symptoms"),
    11: ("1st Deg AV Block", "Low Risk"),
    12: ("2nd AVB/Mobitz I", "Mild Symptoms"),
    13: ("Atrial Flutter", "Severe Symptoms"),
    14: ("Sinus Arrhythmia", "Low Risk"),
    15: ("Other/Remarks", "Low Risk"),
    16: ("Multifocal Atr Tachy", "Severe Symptoms"),
    17: ("Junctional", "Severe Symptoms"),
    18: ("Asystole", "Critical Condition"),
    19: ("Idioventricular", "Severe Symptoms"),
    20: ("Comp Heart Block", "Critical Condition"),
    21: ("Wandering Atrial Pace", "Mild Symptoms"),
    22: ("Paroxysmal Atr Tachy", "Severe Symptoms"),
    23: ("Paced", "Low Risk"),
    24: ("2nd AVB Mobitz II", "Critical Condition"),
    25: ("Zoll Paced", "Mild Symptoms"),
}

hr_filtered["RISK_CLASS"] = hr_filtered["VALUE_MAPPED"].map(lambda x: risk_mapping.get(x, (None, None))[1])
with open(last_layer_file, "r") as file:
    last_layer_embeddings = json.load(file)

# combine features
combined_features = []
labels = []
for _, row in note_sampled.iterrows():
    subject_id = row["SUBJECT_ID"]
    note_emb = row["EMBEDDING"]
    hr_row = hr_filtered[hr_filtered["SUBJECT_ID"] == subject_id]
    if not hr_row.empty:
        hr_emb = hr_row.iloc[0]["EMBEDDING"]
        combined_features.append(np.concatenate([note_emb, hr_emb]))
        labels.append(hr_row.iloc[0]["RISK_CLASS"])

y_combined_encoded = encoder.fit_transform(labels)
if len(hr_filtered) > len(last_layer_embeddings):
    print(f"Warning: HR dataset length ({len(hr_filtered)}) exceeds embeddings length ({len(last_layer_embeddings)}).")
    hr_filtered = hr_filtered.iloc[:len(last_layer_embeddings)].reset_index(drop=True)
elif len(last_layer_embeddings) > len(hr_filtered):
    print(f"Warning: Embeddings length ({len(last_layer_embeddings)}) exceeds HR dataset length ({len(hr_filtered)}).")
    last_layer_embeddings = last_layer_embeddings[:len(hr_filtered)]
hr_filtered["EMBEDDING"] = last_layer_embeddings

'''
# balance
#target_samples_per_class = 300
balanced_hr_data = hr_filtered.groupby("RISK_CLASS", group_keys=False).apply(
    lambda x: x.sample(n=target_samples_per_class, replace=len(x) < target_samples_per_class, random_state=42)
).reset_index(drop=True)
'''
# encode label
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(balanced_hr_data["RISK_CLASS"])
X = np.vstack(balanced_hr_data["EMBEDDING"])
y = y_encoded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
classifier.fit(X_train, y_train)

# eval
y_pred = classifier.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
predictions_df = pd.DataFrame({
    "True_Label": encoder.inverse_transform(y_test),
    "Predicted_Label": encoder.inverse_transform(y_pred)})
predictions_df.to_csv("hr_classification_results_lr.csv", index=False)




