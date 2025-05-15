import pandas as pd
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001

class LSTMDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.tensor([self.targets[idx]], dtype=torch.float32)

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
        y = data[i+seq_len][0]  # Cible uniquement la valeur 'close' future
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Utilisation de l'appareil : {device}")
    logging.info(f"Configuration - SEQ_LEN: {SEQ_LEN}, BATCH_SIZE: {BATCH_SIZE}, EPOCHS: {EPOCHS}, LR: {LR}")
    df = pd.read_parquet("data/SAF_PA_clean.parquet")
    df = df[['close', 'ma_5', 'ma_20', 'return_1d']].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    X, y = create_sequences(scaled, SEQ_LEN)
    # Split train/test
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    train_loader = DataLoader(LSTMDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(LSTMDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(input_size=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Early stopping variables
    best_loss = float('inf')
    patience = 10
    wait = 0

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for seq, target in train_loader:
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.6f}")
        logging.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.6f}")
        # Early stopping check
        if total_loss < best_loss:
            best_loss = total_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logging.info("Early stopping triggered.")
                print("🛑 Early stopping triggered.")
                break

    # Évaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for seq, target in test_loader:
            seq, target = seq.to(device), target.to(device)
            output = model(seq)
            loss = criterion(output, target)
            test_loss += loss.item()
    print(f"📊 Test Loss: {test_loss:.6f}")
    logging.info(f"Test Loss: {test_loss:.6f}")
    # Sauvegarde du modèle
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/lstm_safran.pth")
    print("✅ Modèle sauvegardé sous models/lstm_safran.pth")
    logging.info("Modèle sauvegardé sous models/lstm_safran.pth")

if __name__ == "__main__":
    main()
