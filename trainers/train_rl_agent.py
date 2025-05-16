

import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trainers.env_safran_rl import SafranTradingEnv

# Charger les données de marché propres
df = pd.read_parquet("data/SAF_PA_clean.parquet")

# Créer l'environnement personnalisé Gym
env = DummyVecEnv([lambda: SafranTradingEnv(df)])

# Créer le modèle PPO
model = PPO("MlpPolicy", env, verbose=1)

# Entraîner le modèle
model.learn(total_timesteps=100_000)

# Sauvegarder le modèle
os.makedirs("models", exist_ok=True)
model.save("models/ppo_safran_trader")

print("✅ Agent PPO entraîné et sauvegardé sous models/ppo_safran_trader")