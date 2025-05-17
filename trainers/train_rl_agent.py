import pandas as pd
import os
import sys
import logging
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Logger
logging.basicConfig(
    filename='training_rl.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Setup import local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trainers.env_safran_rl import SafranTradingEnv

# 📦 Config
MODEL_PATH = "models/ppo_safran_trader"
TOTAL_TIMESTEPS = 100_000

# 📈 Charger les données
df = pd.read_parquet("data/SAF_PA_clean.parquet")

# 🔄 Environnement Gym personnalisé
env = DummyVecEnv([lambda: SafranTradingEnv(df)])

# 🎯 Device
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"📟 Device utilisé : {device}")
print(f"📟 Device utilisé : {device}")

# ⚙️ Charger ou créer le modèle
if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH, env=env, device=device, verbose=1)
    logging.info("📥 Modèle PPO existant chargé")
else:
    model = PPO("MlpPolicy", env, verbose=1, device=device)
    logging.info("🆕 Nouveau modèle PPO initialisé")

# 🧠 Entraînement
model.learn(total_timesteps=TOTAL_TIMESTEPS)
logging.info(f"✅ Entraînement terminé : {TOTAL_TIMESTEPS} steps")

# 💾 Sauvegarde
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
logging.info(f"💾 Modèle sauvegardé : {MODEL_PATH}")
