import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

# Configuration
SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001

class LSTMDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    df = pd.read_parquet("data/SAF_PA_clean.parquet")
    df = df[['close', 'ma_5', 'ma_20', 'return_1d']].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    X, y = create_sequences(scaled, SEQ_LEN)
    train_loader = DataLoader(LSTMDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel(input_size=X.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for seq, target in train_loader:
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.6f}")

    torch.save(model.state_dict(), "models/lstm_safran.pth")
    print("✅ Modèle sauvegardé sous models/lstm_safran.pth")

if __name__ == "__main__":
    main()
