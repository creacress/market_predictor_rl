import pandas as pd
import torch
import os
from stable_baselines3 import PPO
from trainers.env_safran_rl import SafranTradingEnv

# Charger les données
df = pd.read_parquet("data/SAF_PA_clean.parquet")

# Charger le modèle entraîné
model_path = "models/ppo_safran_trader.zip"
assert os.path.exists(model_path), "Modèle PPO introuvable, entraînez-le d'abord."
model = PPO.load(model_path)

# Créer un nouvel environnement pour la simulation
env = SafranTradingEnv(df)
obs = env.reset()

done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()

profit = env.balance + env.shares_held * df.iloc[env.current_step]['close'] - env.initial_balance
print(f"\n💰 Profit final simulé : {profit:.2f} €")
