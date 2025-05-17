import pandas as pd
import sys
import os
import logging
import matplotlib.pyplot as plt
import torch
import joblib
from stable_baselines3 import PPO
from trainers.lstm_trainer import LSTMModel, SEQ_LEN

# Ajout du chemin pour les imports locaux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trainers.env_safran_rl import SafranTradingEnv

# Logger
logging.basicConfig(
    filename="simulation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

df = pd.read_parquet("data/SAF_PA_clean.parquet")

# Charger le scaler et le modèle LSTM
scaler = joblib.load("models/lstm_scaler.save")
lstm_model = LSTMModel(input_size=df.shape[1] - 1)  # exclude date if still present
lstm_model.load_state_dict(torch.load("models/lstm_safran.pth", map_location="cpu"))
lstm_model.eval()

# Charger le modèle
model_path = "models/ppo_safran_trader.zip"
assert os.path.exists(model_path), "Modèle PPO introuvable, entraînez-le d'abord."
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PPO.load(model_path, device=device)

# Environnement
env = SafranTradingEnv(df)
obs = env.reset()

done = False
step = 0
profits = []
actions_log = []

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()

    current_price = df.iloc[env.current_step]['close']
    total_value = env.balance + env.shares_held * current_price
    profits.append(total_value)

    # Prédiction LSTM à chaque étape
    if env.current_step >= SEQ_LEN:
        seq = df.iloc[env.current_step - SEQ_LEN:env.current_step][['close', 'ma_5', 'ma_20', 'return_1d', 'rsi']].dropna().to_numpy()
        seq_scaled = scaler.transform(seq)
        input_seq = torch.FloatTensor(seq_scaled).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = lstm_model(input_seq).item()
        close_min, close_max = scaler.data_min_[0], scaler.data_max_[0]
        predicted_price = pred_scaled * (close_max - close_min) + close_min
    else:
        predicted_price = None

    action_label = ['HOLD', 'BUY', 'SELL'][action]
    logging.info(f"Step {step} | Action: {action_label} | Reward: {reward:.2f}")
    actions_log.append({
        "step": step,
        "action": action_label,
        "price": current_price,
        "capital": total_value,
        "reward": reward,
        "lstm_pred": predicted_price
    })

    if predicted_price:
        print(f"Step {step} | LSTM prédit : {predicted_price:.2f} €, Prix réel : {current_price:.2f}")

    step += 1

# Résultat final
profit = env.balance + env.shares_held * df.iloc[env.current_step]['close'] - env.initial_balance
summary = f"💰 Profit final simulé : {profit:.2f} €"
print("\n" + summary)
logging.info(summary)

# Enregistrer les actions
pd.DataFrame(actions_log).to_csv("predict/actions_log.csv", index=False)

# Graphique capital avec points BUY/SELL
plt.figure(figsize=(12, 6))
plt.plot(profits, label="Capital total")

for entry in actions_log:
    if entry['action'] == 'BUY':
        plt.scatter(entry['step'], entry['capital'], color='green', label='BUY' if 'BUY' not in plt.gca().get_legend_handles_labels()[1] else "")
    elif entry['action'] == 'SELL':
        plt.scatter(entry['step'], entry['capital'], color='red', label='SELL' if 'SELL' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Évolution du capital de l'agent PPO avec décisions")
plt.xlabel("Jour")
plt.ylabel("Capital (€)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("predict/profits_plot.png")
plt.show()
logging.info("📈 Graphique enregistré : profits_plot.png")