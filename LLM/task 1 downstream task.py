import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import matplotlib.pyplot as plt
import matplotlib

# train the classifier
features_file = "gpt2_features.json"
features_data = pd.read_json(features_file, lines=True)

X = np.array(features_data["HIDDEN_STATES"].tolist())
y = features_data["CLASSIFICATION"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# get the report
y_pred = classifier.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))


# apply on local data
local_data = pd.read_csv("D:\\homework\\AI Healthcare\\CSV\\blood_sugar_tests.csv", usecols=["SUBJECT_ID", "VALUE"])
local_data = local_data.dropna(subset=["VALUE"])
local_data["VALUE"] = pd.to_numeric(local_data["VALUE"], errors="coerce")
local_data = local_data.dropna(subset=["VALUE"])

# load gpt2 tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
model = GPT2LMHeadModel.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def create_prompt(row):
    return f"""Subject ID: {row['SUBJECT_ID']}Blood Sugar Level: {row['VALUE']} mg/dL"""

# get hidden states for local data
hidden_states_local = []
for idx, row in local_data.iterrows():
    prompt = create_prompt(row)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_layer_hidden_states = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
    hidden_states_local.append(last_layer_hidden_states)


X_local = np.array(hidden_states_local)
local_data["PREDICTED_CLASSIFICATION"] = classifier.predict(X_local)

# save the results
output_file = "local_data_classification_results.csv"
local_data.to_csv(output_file, index=False)



# visualization
classification_counts = local_data["PREDICTED_CLASSIFICATION"].value_counts()
data_local = pd.read_csv('D:\\homework\\AI Healthcare\\LLM\\local_data_classification_results.csv')
#print(data_local.info())
matplotlib.use('TkAgg')


# pie graph
plt.figure(figsize=(8, 8))
plt.pie(
    classification_counts,
    labels=classification_counts.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Diabetes Predicated Classification")
plt.tight_layout()
plt.savefig("diabetes classification.svg", format="svg")
plt.show()

# box plot
data_local_filtered = data_local[
    ((data_local["PREDICTED_CLASSIFICATION"] == "Low Risk") & (data_local["VALUE"] < 140)) |
    ((data_local["PREDICTED_CLASSIFICATION"] == "Risk") & (data_local["VALUE"].between(140, 199))) |
    ((data_local["PREDICTED_CLASSIFICATION"] == "High Risk") & (data_local["VALUE"] >= 200))
]
classifications = ["Low Risk", "Risk", "High Risk"]
classification_data = {
    classification: data_local_filtered[data_local_filtered["PREDICTED_CLASSIFICATION"] == classification]["VALUE"]
    for classification in classifications
}

plt.figure(figsize=(10, 6))
plt.boxplot(
    [classification_data[classification] for classification in classifications],
    vert=False,
    patch_artist=True,
    showfliers=False,
    labels=classifications
)

plt.title("Box Plot of Predicted Classification", fontsize=14)
plt.xlabel("Blood Sugar Level (mg/dL)")
plt.ylabel("Classification")
plt.tight_layout()
plt.savefig("box plot classification.svg", format="svg")
plt.show()

# summary
summary_stats = data_local.groupby("PREDICTED_CLASSIFICATION")["VALUE"].agg(
    mean="mean",
    median="median",
    first_quartile=lambda x: x.quantile(0.25),
    third_quartile=lambda x: x.quantile(0.75)
).reset_index()
print(summary_stats)

