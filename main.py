


import pandas as pd
import torch
import joblib
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trainers.env_safran_rl import SafranTradingEnv
from trainers.lstm_trainer import LSTMModel, SEQ_LEN

logging.basicConfig(
    filename='main.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 1. Charger les données
df = pd.read_parquet("data/SAF_PA_clean.parquet")
print("✅ Données chargées.")
logging.info("Données chargées depuis data/SAF_PA_clean.parquet")

# 2. Initialiser l'environnement avec LSTM intégré
env = DummyVecEnv([lambda: SafranTradingEnv(df)])
print("🔗 Environnement PPO + LSTM prêt.")
logging.info("Environnement PPO + LSTM initialisé.")

# 3. Initialiser le modèle PPO
model = PPO("MlpPolicy", env, verbose=1)
print("🧠 PPO initialisé.")
logging.info("Modèle PPO initialisé.")

# 4. Entraînement
print("🚀 Entraînement du modèle...")
logging.info("Début de l'entraînement PPO.")
model.learn(total_timesteps=100_000)
print("✅ Entraînement terminé.")
logging.info("Entraînement PPO terminé.")

# 5. Sauvegarde
model.save("models/ppo_safran_trader")
print("💾 Modèle sauvegardé.")
logging.info("Modèle PPO sauvegardé dans models/ppo_safran_trader.")

# 6. Prédiction LSTM du dernier jour pour affichage
scaler = joblib.load("models/lstm_scaler.save")
df_for_lstm = df[['close', 'ma_5', 'ma_20', 'return_1d', 'rsi']].dropna()
scaled = scaler.transform(df_for_lstm.values)
input_seq = torch.FloatTensor(scaled[-SEQ_LEN:]).unsqueeze(0)

lstm_model = LSTMModel(input_size=input_seq.shape[2])
lstm_model.load_state_dict(torch.load("models/lstm_safran.pth", map_location="cpu"))
lstm_model.eval()

with torch.no_grad():
    pred_scaled = lstm_model(input_seq).item()

close_min, close_max = scaler.data_min_[0], scaler.data_max_[0]
predicted_price = pred_scaled * (close_max - close_min) + close_min

print(f"📈 Prédiction LSTM (prochaine clôture estimée) : {predicted_price:.2f} €")
logging.info(f"Prédiction LSTM du prochain close : {predicted_price:.2f} €")