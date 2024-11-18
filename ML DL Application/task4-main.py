import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

lab = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\lab_with_labels.csv', low_memory=False)
pre = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\pre_pat.csv', low_memory=False)
item = pd.read_csv('D:\\homework\\AI Healthcare\\CSV\\new\\d_labitems.csv', low_memory=False)


top_20_drugs = pre['drug'].value_counts().head(20).index.tolist()
pre_top_20 = pre[pre['drug'].isin(top_20_drugs)].copy()
pre_top_20['drug_idx'] = LabelEncoder().fit_transform(pre_top_20['drug'])


binary_labels = pre_top_20.groupby('subject_id')['drug_idx'].apply(
    lambda x: np.bincount(x, minlength=20)
).apply(pd.Series).reset_index()
binary_labels.columns = ['subject_id'] + [f'drug_{i}' for i in range(20)]


lab['value'] = pd.to_numeric(lab['value'], errors='coerce')
lab = lab.dropna(subset=['value'])
lab['value_normalized'] = MinMaxScaler().fit_transform(lab[['value']])


lab['itemid_encoded'] = LabelEncoder().fit_transform(lab['itemid'])
data = lab.merge(binary_labels, on='subject_id', how='inner')
grouped_data = data.groupby('subject_id').agg({
    'itemid_encoded': list,
    'value_normalized': list,
    **{f'drug_{i}': 'max' for i in range(20)}
}).reset_index()

for col in [f'drug_{i}' for i in range(20)]:
    grouped_data[col] = pd.to_numeric(grouped_data[col], errors='coerce')
grouped_data = grouped_data.dropna(subset=[f'drug_{i}' for i in range(20)]).reset_index(drop=True)


train, test = train_test_split(grouped_data, test_size=0.2, random_state=42)
for col in [f'drug_{i}' for i in range(20)]:
    grouped_data[col] = pd.to_numeric(grouped_data[col], errors='coerce')
grouped_data = grouped_data.dropna(subset=[f'drug_{i}' for i in range(20)]).reset_index(drop=True)


# define dataset
class DrugPredictionDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        item_ids = torch.tensor(row['itemid_encoded'][:self.max_seq_len], dtype=torch.long)
        values = torch.tensor(row['value_normalized'][:self.max_seq_len], dtype=torch.float32)
        if len(item_ids) < self.max_seq_len:
            padding_len = self.max_seq_len - len(item_ids)
            item_ids = torch.cat([item_ids, torch.zeros(padding_len, dtype=torch.long)])
            values = torch.cat([values, torch.zeros(padding_len, dtype=torch.float32)])
        labels = torch.tensor(row[[f'drug_{i}' for i in range(20)]].values.astype(float), dtype=torch.float32)

        return item_ids, values, labels



max_seq_len = 100
train_dataset = DrugPredictionDataset(train, max_seq_len)
test_dataset = DrugPredictionDataset(test, max_seq_len)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# define the model, more complex than i expected
class MultiTaskContrastiveModel(nn.Module):
    def __init__(self, num_items, embedding_dim, num_drugs):
        super(MultiTaskContrastiveModel, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.value_encoder = nn.Linear(1, embedding_dim)
        self.fusion_layer = nn.Linear(embedding_dim * 2, embedding_dim)
        self.attention = nn.Linear(embedding_dim, 1)
        self.shared_encoder = nn.Transformer(d_model=embedding_dim, nhead=4, num_encoder_layers=2)
        self.output_heads = nn.ModuleList([nn.Linear(embedding_dim, 1) for _ in range(num_drugs)])

    def forward(self, item_ids, values):
        item_embeds = self.item_embedding(item_ids)
        value_embeds = self.value_encoder(values.unsqueeze(-1))
        fused = torch.cat([item_embeds, value_embeds], dim=-1)
        fused = self.fusion_layer(fused)
        attention_weights = F.softmax(self.attention(fused), dim=1)
        weighted_fused = fused * attention_weights
        shared_features = self.shared_encoder(weighted_fused, weighted_fused)
        shared_features = shared_features.mean(dim=1)
        outputs = [torch.sigmoid(head(shared_features)) for head in self.output_heads]
        return torch.cat(outputs, dim=-1)

    def get_embeddings(self, item_ids, values):
        item_embeds = self.item_embedding(item_ids)
        value_embeds = self.value_encoder(values.unsqueeze(-1))
        fused = torch.cat([item_embeds, value_embeds], dim=-1)
        fused = self.fusion_layer(fused)
        attention_weights = F.softmax(self.attention(fused), dim=1)
        weighted_fused = fused * attention_weights

        shared_features = self.shared_encoder(weighted_fused,
                                              weighted_fused)
        return shared_features.mean(dim=1)


# contrastive loss
def contrastive_loss(embeddings, labels, margin=1.0):
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    label_similarity = torch.mm(labels, labels.T) > 0
    label_similarity = label_similarity.float()

    pos_pairs = label_similarity * (1 - similarity_matrix) ** 2
    neg_pairs = (1 - label_similarity) * torch.clamp(similarity_matrix - margin, min=0) ** 2
    loss = torch.mean(pos_pairs + neg_pairs)
    return loss



# training part
model = MultiTaskContrastiveModel(num_items=len(lab['itemid_encoded'].unique()), embedding_dim=64, num_drugs=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
bce_loss = nn.BCELoss()
epoch_num = 10
model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)
'''
for epoch in range(epoch_num):
    model.train()
    epoch_bce_loss = 0
    epoch_contrastive_loss = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}", unit="batch") as pbar:
        for item_ids, values, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(item_ids, values)
            embeddings = model.get_embeddings(item_ids, values)

            bce = bce_loss(predictions, labels) # compute the loss
            contrastive = contrastive_loss(embeddings, labels) #compute the con loss
            total_loss = bce + 0.1 * contrastive

            total_loss.backward()
            optimizer.step()
            epoch_bce_loss += bce.item()
            epoch_contrastive_loss += contrastive.item()
            pbar.set_postfix({"BCE Loss": bce.item(), "Contrastive Loss": contrastive.item()})
            pbar.update(1)
    print(f"Epoch {epoch + 1} Summary: "
          f"BCE Loss = {epoch_bce_loss / len(train_loader):.4f}, "
          f"Contrastive Loss = {epoch_contrastive_loss / len(train_loader):.4f}")

    # save the model every two epochs
    if (epoch + 1) % 2 == 0:
        model_save_path = os.path.join(model_save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
'''
# evaluation
saved_model_path = os.path.join(model_save_dir, f"model_epoch_{epoch_num}.pth")
model.load_state_dict(torch.load(saved_model_path))
model.eval()
def evaluate_model_with_report(model, test_loader, num_drugs=20, drug_names=None):
    if drug_names is None:
        drug_names = [f"drug_{i}" for i in range(num_drugs)]
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for item_ids, values, labels in test_loader:
            predictions = model(item_ids, values)
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    binary_predictions = (all_predictions > 0.5).astype(int)
    all_labels = np.clip(all_labels, 0, 1).astype(int)
    binary_predictions = np.clip(binary_predictions, 0, 1).astype(int)
    drug_precision = []
    drug_recall = []
    drug_f1 = []
    for i in range(num_drugs):
        precision = precision_score(all_labels[:, i], binary_predictions[:, i], average='binary', zero_division=0)
        recall = recall_score(all_labels[:, i], binary_predictions[:, i], average='binary', zero_division=0)
        f1 = f1_score(all_labels[:, i], binary_predictions[:, i], average='binary', zero_division=0)
        drug_precision.append(precision)
        drug_recall.append(recall)
        drug_f1.append(f1)

    # this part print the same thing. LOL
    print("Per-Drug Evaluation Report:")
    print(f"{'Drug':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)
    for i in range(num_drugs):
        print(f"{drug_names[i]:<20} {drug_precision[i]:<10.4f} {drug_recall[i]:<10.4f} {drug_f1[i]:<10.4f}")

    avg_precision = np.mean(drug_precision)
    avg_recall = np.mean(drug_recall)
    avg_f1 = np.mean(drug_f1)
    print("\nAverage Metrics:")
    print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-Score: {avg_f1:.4f}")

    # detailed classification report
    print("\nDetailed Classification Report:")
    print(
        classification_report(
            all_labels,
            binary_predictions,
            target_names=drug_names,
            zero_division=0
        )
    )
evaluate_model_with_report(model, test_loader, num_drugs=20, drug_names=top_20_drugs)


# for t-sne
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Generate embeddings for the test set
model.eval()
all_embeddings = []
all_labels = []
with torch.no_grad():
    for item_ids, values, labels in test_loader:
        embeddings = model.get_embeddings(item_ids, values)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Stack the embeddings and labels
all_embeddings = np.vstack(all_embeddings)
all_labels = np.vstack(all_labels)

# Apply t-SNE to reduce embeddings to 2D for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(all_embeddings)

# Assign a unique color to each drug
drug_colors = [f"C{i}" for i in range(len(top_20_drugs))]

# Visualize the embeddings
plt.figure(figsize=(12, 8))
for i in range(len(top_20_drugs)):
    indices = np.where(all_labels[:, i] == 1)[0]  # Find indices for the current drug
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                label=top_20_drugs[i], alpha=0.6, c=drug_colors[i])

plt.title("t-SNE Visualization of Embeddings", fontsize=16)
plt.xlabel("t-SNE Component 1", fontsize=12)
plt.ylabel("t-SNE Component 2", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig("t-SNE-task4.svg", format="svg")
plt.show()
