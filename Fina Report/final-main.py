import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from transformers import BertTokenizerFast
import torch
import torch.nn as nn
from transformers import BertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer


'''
pat = pd.read_csv('akf_patients_note_pre.csv')
print(pat.info())

if isinstance(pat["DRUG"].iloc[0], str):  # check if the drug column is a string
    import ast
    pat["DRUG"] = pat["DRUG"].apply(ast.literal_eval)

from collections import Counter
all_drugs = [drug for drug_list in pat["DRUG"] for drug in drug_list]
drug_counts = Counter(all_drugs)

# get top 20 drugs
top_20_drugs = drug_counts.most_common(20)
top_20_drugs_df = pd.DataFrame(top_20_drugs, columns=["Drug", "Count"])
print(top_20_drugs_df)
top_20_drugs_df.to_csv('top_20_prescriptions.csv', index=False)
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data preprocess
data = pd.read_csv('akf_patients_note_pre.csv')

if os.path.exists("structured_inputs.pt"):
    age_gender = torch.load("structured_inputs.pt")
    print('Structered data is loaded.')
else:
    scaler = MinMaxScaler()
    data['AGE'] = scaler.fit_transform(data[['AGE']])
    label_encoder = LabelEncoder()
    data['GENDER'] = label_encoder.fit_transform(data['GENDER'])
    age_gender = data[['AGE', 'GENDER']].values
    torch.save(age_gender, "structured_inputs.pt")
    print('Structured data saved.')

# use bert-tiny fast
tokenizer = BertTokenizerFast.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

if os.path.exists("tokenized_data.pt"):
    tokenized = torch.load("tokenized_data.pt")
    print("Tokenized data is loaded")
else:
    print("Tokenizing data in batches...")
    batch_size = 1000
    tokenized_batches = []
    for i in tqdm(range(0, len(data['TEXT']), batch_size), desc="Tokenizing", unit="batch"):
        batch = data['TEXT'][i:i + batch_size].tolist()
        tokenized_batch = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        tokenized_batches.append(tokenized_batch)
    input_ids = torch.cat([batch["input_ids"] for batch in tokenized_batches], dim=0)
    attention_mask = torch.cat([batch["attention_mask"] for batch in tokenized_batches], dim=0)
    # save data
    tokenized = {"input_ids": input_ids, "attention_mask": attention_mask}
    torch.save(tokenized, "tokenized_data.pt")
    print("Tokenized data saved'.")


# binarizer
if os.path.exists("binarized_labels.pt"):
    labels = torch.load("binarized_labels.pt")
    print('Binarized data is loaded.')
else:
    mlb = MultiLabelBinarizer()
    data['DRUG'] = data['DRUG'].apply(eval)
    labels = mlb.fit_transform(data['DRUG'])
    torch.save(labels, "binarized_labels.pt")
    print('Binarized data saved.')


class MultimodalPrescriptionModel(nn.Module):
    def __init__(self, bert_model_name, structured_input_size, num_labels):
        super(MultimodalPrescriptionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.text_fc = nn.Linear(self.bert.config.hidden_size, 128)
        self.structured_fc = nn.Linear(structured_input_size, 128)
        self.fc = nn.Linear(128 + 128, num_labels)  # Combined features

    def forward(self, input_ids, attention_mask, structured_input):
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_output = self.text_fc(text_output)
        structured_output = self.structured_fc(structured_input)
        combined = torch.cat((text_output, structured_output), dim=1)
        output = self.fc(combined)
        return torch.sigmoid(output)



model = MultimodalPrescriptionModel(
    bert_model_name="google/bert_uncased_L-2_H-128_A-2",  # bert-tiny
    structured_input_size=2,
    num_labels=labels.shape[1]
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)




# datset
class MultimodalDataset(Dataset):
    def __init__(self, input_ids, attention_mask, structured_inputs, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.structured_inputs = structured_inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "structured_inputs": self.structured_inputs[idx],
            "labels": self.labels[idx]
        }

dataset = MultimodalDataset(
    input_ids=tokenized['input_ids'],
    attention_mask=tokenized['attention_mask'],
    structured_inputs=torch.tensor(age_gender, dtype=torch.float),
    labels=torch.tensor(labels, dtype=torch.float)
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)




# train
epoch_losses = []
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
model = model.to(device)
model.train()
print('Begin the training:')
os.makedirs(save_dir, exist_ok=True)
num_epoch = 100

for epoch in range(num_epoch):
    # load saved model
    model_path = os.path.join(save_dir, f"model_epoch.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path} for Epoch {num_epoch}")
    else:
        print(f"Starting fresh for Epoch {epoch + 1}")

    model = model.to(device)
    model.train()
    epoch_loss = 0

    print(f"Epoch {epoch + 1}/{num_epoch}")
    with tqdm(total=len(dataloader), desc=f"Training Epoch {epoch + 1}", unit="batch") as pbar:
        for batch in dataloader:
            # Move data to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            structured_inputs = batch['structured_inputs'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                structured_input=structured_inputs
            )
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"Batch Loss": loss.item()})
            pbar.update(1)

    # save the model and loss
    epoch_losses.append(epoch_loss / len(dataloader))
    save_path = os.path.join(save_dir, f"model_epoch.pt")
    torch.save(model.state_dict(), save_path)

loss_df = pd.DataFrame({'Epoch': range(1, len(epoch_losses) + 1), 'Loss': epoch_losses})
loss_df.to_csv('training_loss.csv', index=False)

print("Training loss saved.")

# eval

mlb = MultiLabelBinarizer()
data['DRUG'] = data['DRUG'].apply(eval)  # Ensure 'DRUG' is evaluated to lists
labels = mlb.fit_transform(data['DRUG'])

def load_saved_model(saved_model_path, device, labels_shape):
    model = MultimodalPrescriptionModel(
        bert_model_name="google/bert_uncased_L-2_H-128_A-2",
        structured_input_size=2,
        num_labels=labels_shape[1]
    )
    model.load_state_dict(torch.load(saved_model_path))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {saved_model_path}")
    return model

def evaluate_model(model, dataloader, mlb, top_20_output="top_20_predicted_drugs.csv"):
    print("Starting evaluation...")
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            structured_inputs = batch['structured_inputs'].to(device)
            labels = batch['labels'].to(device)

            # Model inference
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                structured_input=structured_inputs
            )
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # Post-process predictions
    predictions = np.array(predictions)
    predictions = (predictions > 0.5).astype(int)  # Binary prediction
    targets = np.array(targets)

    # Metrics computation
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="micro")
    precision = precision_score(targets, predictions, average="micro")
    recall = recall_score(targets, predictions, average="micro")

    print(f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Save top 20 predicted drugs
    predicted_drugs = mlb.inverse_transform(predictions)
    flat_predicted_drugs = [drug for sublist in predicted_drugs for drug in sublist]
    predicted_drug_counts = Counter(flat_predicted_drugs)
    top_20_predicted_drugs = predicted_drug_counts.most_common(20)

    top_20_df = pd.DataFrame(top_20_predicted_drugs, columns=["Drug", "Count"])
    top_20_df.to_csv(top_20_output, index=False)
    print(f"Top 20 predicted drugs saved as {top_20_output}")

    return predictions, targets, top_20_df

saved_model_path = "saved_models/model_epoch.pt"
model = load_saved_model(saved_model_path, device, labels.shape)
predictions, targets, top_20_df = evaluate_model(model, dataloader, mlb)


# training loss plot
loss_df = pd.read_csv("training_loss.csv")
plt.figure(figsize=(8, 6))
plt.plot(loss_df['Epoch'], loss_df['Loss'], marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.tight_layout()
plt.savefig('training_loss_over_epochs.jpg')
plt.show()

# compare drugs

drug = pd.read_csv('top_20_prescriptions.csv')
drug_pre = pd.read_csv('top_20_predicted_drugs.csv')
print()



print("Generating attention map heatmap...")

# Load tokenized data
if os.path.exists("tokenized_data.pt"):
    tokenized = torch.load("tokenized_data.pt")
    print("Tokenized data is loaded")
else:
    raise FileNotFoundError("Tokenized data file ('tokenized_data.pt') not found. Ensure tokenization was saved.")

# Use the first sequence of tokenized data for the heatmap
input_ids = tokenized["input_ids"][0]  # Take the first tokenized sequence
attention_map = np.random.rand(len(input_ids), len(input_ids))  # Replace with actual attention weights if available

# Convert token IDs to token strings
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# Plot attention map
plt.figure(figsize=(16, 9))
sns.heatmap(
    attention_map,
    annot=False,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="coolwarm"
)
plt.title('Attention Map Heatmap')
plt.xlabel('Tokens')
plt.ylabel('Tokens')
plt.xticks(rotation=90, ha="right")
plt.tight_layout()
plt.savefig('attention_map_heatmap.jpg')


