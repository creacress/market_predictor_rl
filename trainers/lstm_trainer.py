import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import logging
import ta  # pip install ta

logging.basicConfig(
    filename='training_lstm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Config
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
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len][0])  # prédiction sur la colonne 'close'
    return np.array(xs), np.array(ys)

def main():
    df = pd.read_parquet("data/SAF_PA_clean.parquet")

    # Ajouter indicateurs techniques
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['macd_signal'] = ta.trend.MACD(close=df['close']).macd_signal()
    df = df[['close', 'ma_5', 'ma_20', 'return_1d', 'rsi', 'macd', 'macd_signal']].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    joblib.dump(scaler, 'models/lstm_scaler.save')

    X, y = create_sequences(scaled, SEQ_LEN)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    train_loader = DataLoader(LSTMDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(LSTMDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"📟 Device utilisé : {device}")
    model = LSTMModel(input_size=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
        logging.info(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss:.6f}")

    # Test Loss
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for seq, target in test_loader:
            seq, target = seq.to(device), target.to(device)
            output = model(seq)
            loss = criterion(output, target)
            test_loss += loss.item()
    logging.info(f"📊 Test Loss: {test_loss:.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_safran.pth")
    logging.info("✅ Modèle LSTM sauvegardé sous models/lstm_safran.pth")

if __name__ == "__main__":
    main()
