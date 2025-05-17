import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trainers.lstm_trainer import LSTMModel, SEQ_LEN
import yfinance as yf

# Charger les données
df = pd.read_parquet("data/SAF_PA_clean.parquet")
df = df[['close', 'ma_5', 'ma_20', 'return_1d']].dropna()

# Scaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

# Prendre les dernières SEQ_LEN séquences
latest_sequence = scaled_data[-SEQ_LEN:]
input_seq = torch.FloatTensor(latest_sequence).unsqueeze(0)

# Charger le modèle
model = LSTMModel(input_size=input_seq.shape[2])
model.load_state_dict(torch.load("models/lstm_safran.pth"))
model.eval()

# Prédiction
with torch.no_grad():
    predicted_scaled = model(input_seq).item()

# Revenir à l’échelle réelle
dummy_input = np.zeros((1, scaled_data.shape[1]))
dummy_input[0, 0] = predicted_scaled  # seule la colonne 'close'
predicted_real = scaler.inverse_transform(dummy_input)[0, 0]

print(f"📈 Prochaine prédiction de clôture : {predicted_real:.2f} €")

# Comparaison avec la valeur réelle de J-1 (si dispo)
ticker = "SAF.PA"
data = yf.download(ticker, period="2d", interval="1d")
if not data.empty and "Close" in data.columns:
    last_real_close = data["Close"].iloc[-1].item()
    print(f"🔍 Clôture réelle la plus récente : {last_real_close:.2f} €")
    delta = predicted_real - last_real_close
    print(f"🔁 Écart prédiction / réel : {delta:.2f} €")
else:
    print("⚠️ Impossible de récupérer la clôture réelle pour comparaison.")