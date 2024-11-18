import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# load the reduced data
note = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\NOTEEVENTS_reduced.csv', low_memory=False)
adm = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\ADMISSIONS.csv', low_memory=False)

# merge the data with adm

data = pd.merge(note, adm[['HADM_ID', 'ADMISSION_TYPE']], on='HADM_ID')
data = data.dropna(subset=['TEXT', 'ADMISSION_TYPE'])
#print(data.info())

# encode
label_mapping = {
    "EMERGENCY": 0,
    "ELECTIVE": 1,
    "NEWBORN": 2,
    "URGENT": 3
}
data['LABEL'] = data['ADMISSION_TYPE'].map(label_mapping)

# clean the data
data = data.dropna(subset=['LABEL'])
data['LABEL'] = data['LABEL'].astype(int)

print(f"Dataset prepared: {len(data)} rows")
print(data['LABEL'].value_counts())

# data creation
X_train, X_test, y_train, y_test = train_test_split(data['TEXT'], data['LABEL'], test_size=0.2, random_state=42)

class AdmissionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts.iloc[index]
        label = self.labels.iloc[index]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')

# creat the dataset and dataloader
train_dataset = AdmissionDataset(X_train, y_train, tokenizer, max_length=512)
test_dataset = AdmissionDataset(X_test, y_test, tokenizer, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)



# define the model
tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained('google/bert_uncased_L-2_H-128_A-2', num_labels=4)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)


print("Begin to train")
# training
def train_model(model, data_loader, optimizer, device, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        correct = 0
        total = 0
        for batch in tqdm(data_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Train Loss: {epoch_loss / len(data_loader):.4f}, Train Accuracy: {correct / total:.4f}")

train_model(model, train_loader, optimizer, device)

# evaluation
print("Begin to evaluate")
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_mapping.keys()))

evaluate_model(model, test_loader, device)
