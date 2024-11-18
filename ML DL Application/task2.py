import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import numpy as np




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

note = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\NOTEEVENTS_reduced.csv', low_memory=False)
adm = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\ADMISSIONS.csv', low_memory=False)

data = pd.merge(note, adm[['HADM_ID', 'ADMISSION_TYPE']], on='HADM_ID')
data = data.dropna(subset=['TEXT', 'ADMISSION_TYPE'])

label_mapping = {
    "EMERGENCY": 0,
    "ELECTIVE": 1,
    "NEWBORN": 2,
    "URGENT": 3
}
data['LABEL'] = data['ADMISSION_TYPE'].map(label_mapping)
data = data.dropna(subset=['LABEL'])
data['LABEL'] = data['LABEL'].astype(int)
print(f"Dataset prepared: {len(data)} rows")
print(data['LABEL'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(data['TEXT'], data['LABEL'], test_size=0.2, random_state=42)


# define dataset clss
class NoteDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# define dataloader
model_name = "google/bert_uncased_L-2_H-128_A-2"  # we use BERT-Tiny model like in task 1
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = NoteDataset(X_train, y_train, tokenizer)
test_dataset = NoteDataset(X_test, y_test, tokenizer)
batch_num = 512
train_loader = DataLoader(train_dataset, batch_size=batch_num, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_num)


# define the contrastive learning model
class ContrastiveLearningModel(nn.Module):
    def __init__(self, model_name, embedding_dim):
        super(ContrastiveLearningModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output
        projections = self.projection(embeddings)
        return embeddings, projections

embedding_dim = 128
model = ContrastiveLearningModel(model_name, embedding_dim).to(device)


# contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings_a, embeddings_b, labels):
        distances = torch.norm(embeddings_a - embeddings_b, dim=1)
        labels = labels.float()
        loss = (labels * distances.pow(2) + (1 - labels) * torch.clamp(self.margin - distances, min=0).pow(2)).mean()
        return loss


criterion = ContrastiveLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

# train the model
epochs = 10
print("Begin training...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        embeddings, projections = model(input_ids, attention_mask)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        loss = criterion(projections[:-1], projections[1:], labels[:-1] == labels[1:])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")


model.eval()
embeddings = []
labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"]

        embedding, _ = model(input_ids, attention_mask)
        embeddings.append(embedding.cpu().numpy())
        labels.extend(label.numpy())

embeddings = np.vstack(embeddings)
labels = np.array(labels)

# apply logistic regression classifier as evaluation
X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
classifier = LogisticRegression(max_iter=10000)
classifier.fit(X_train_emb, y_train_emb)
y_pred = classifier.predict(X_test_emb)
print("Classification Report:")
print(classification_report(y_test_emb, y_pred, target_names=label_mapping.keys()))

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
for label in np.unique(labels):
    plt.scatter(
        embeddings_2d[labels == label, 0],
        embeddings_2d[labels == label, 1],
        label=f"Class {label}",
        alpha=0.6
    )
plt.legend()
plt.title("t-SNE Visualization of Embeddings")
plt.tight_layout()
plt.savefig("t-SNE.svg", format="svg")
plt.show()


