import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

train_file = "file.csv"
val_file = "file.csv"
test_file = "file.csv"
attribute = "relevance"

print(train_file)
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, _ = self.lstm(embedded)
        hidden = torch.cat((output[:, -1, :hidden_dim], output[:, 0, hidden_dim:]), dim=1)
        return self.fc(hidden)

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_fn(batch):
    inputs, labels = zip(*batch)
    # Pad sequences to the same length
    padded_inputs = pad_sequence([torch.LongTensor(seq) for seq in inputs], batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return padded_inputs, labels


def merge_patch_msg(df):
    def merge_msg_patch(row):
        return row['msg'] + ' [SEP] ' + row['patch']


    df['merged'] = df.apply(merge_msg_patch, axis=1)

    return df

def preprocess_msg(text):
    return text


train_data = pd.read_csv(train_file)
valid_data = pd.read_csv(val_file)
test_data = pd.read_csv(test_file)


if attribute == "informativeness" or attribute == "expression":
    train_data['msg'] = train_data['msg'].apply(preprocess_msg)
    valid_data['msg'] = valid_data['msg'].apply(preprocess_msg)
    test_data['msg'] = test_data['msg'].apply(preprocess_msg)
else:
    merged_train_data = merge_patch_msg(train_data)
    train_data['msg'] = merged_train_data['merged'].apply(preprocess_msg)



label_encoder = LabelEncoder()
train_data[attribute] = label_encoder.fit_transform(train_data[attribute])
valid_data[attribute] = label_encoder.transform(valid_data[attribute])
test_data[attribute] = label_encoder.transform(test_data[attribute])


all_words = [word for msg in train_data['msg'] for word in msg.split()]
word_counts = Counter(all_words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
vocab_to_int = {word: idx + 1 for idx, word in enumerate(vocab)}
vocab_size = len(vocab_to_int) + 1


train_data['msg'] = train_data['msg'].apply(lambda x: [vocab_to_int[word] for word in x.split()])
valid_data['msg'] = valid_data['msg'].apply(lambda x: [vocab_to_int.get(word, 0) for word in x.split()])
test_data['msg'] = test_data['msg'].apply(lambda x: [vocab_to_int.get(word, 0) for word in x.split()])


embedding_dim = 100
hidden_dim = 128
output_dim = len(label_encoder.classes_)
num_layers = 2
bidirectional = True
dropout = 0.2
batch_size = 64
epochs = 10
learning_rate = 0.001


train_dataset = TextDataset(train_data['msg'].values, train_data[attribute].values)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dataset = TextDataset(valid_data['msg'].values, valid_data[attribute].values)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_dataset = TextDataset(test_data['msg'].values, test_data[attribute].values)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)


model = BiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs = torch.LongTensor(inputs)  
        labels = torch.LongTensor(labels)  
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Evaluate on validation data
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = torch.LongTensor(inputs)  
            labels = torch.LongTensor(labels)  
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    valid_loss /= len(valid_loader)
    accuracy = correct / total
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}')


true_labels = []
predicted_labels = []


model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = torch.LongTensor(inputs)
        labels = torch.LongTensor(labels)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())


accuracy = accuracy_score(true_labels, predicted_labels)
balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f'Accuracy: {accuracy:.4f}')
print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')