import os
import time
import pandas as pd
import logging
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trainers.env_safran_rl import SafranTradingEnv

# Logger config
logging.basicConfig(
    filename='training_continuous.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODEL_PATH = "models/ppo_safran_trader"

while True:
    logging.info("🔁 Lancement de l'entraînement continu")

    # Charger les données récentes (ex: 60 derniers jours)
    df = pd.read_parquet("data/SAF_PA_clean.parquet")
    df = df.tail(60)

    # Créer environnement
    env = DummyVecEnv([lambda: SafranTradingEnv(df)])

    # Charger ou créer le modèle
    if os.path.exists(MODEL_PATH + ".zip"):
        model = PPO.load(MODEL_PATH, env=env, verbose=1)
        logging.info("📥 Modèle PPO existant chargé")
    else:
        model = PPO("MlpPolicy", env, verbose=1)
        logging.info("🆕 Nouveau modèle PPO initialisé")

    # Réentraînement
    model.learn(total_timesteps=10_000)
    model.save(MODEL_PATH)
    logging.info("✅ Modèle mis à jour et sauvegardé")

    # Pause (ex: 1 jour = 86400s, pour test = 60s)
    logging.info("⏱️ Pause avant prochaine itération\n")
    time.sleep(86400)  # pour test : remplace par 60
