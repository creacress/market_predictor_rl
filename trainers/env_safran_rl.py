import gym
import numpy as np
from gym import spaces
import logging
import ta
import torch
import joblib
from trainers.lstm_trainer import LSTMModel, SEQ_LEN

logging.basicConfig(
    filename='training_env.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SafranTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, window_size=10):
        super(SafranTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.df["day"] = self.df["date"].dt.day
        self.df["month"] = self.df["date"].dt.month
        self.df["weekday"] = self.df["date"].dt.weekday
        self.df.drop(columns=["date"], inplace=True)
        self.df["rsi"] = ta.momentum.RSIIndicator(close=self.df["close"]).rsi()
        self.df.dropna(inplace=True)

        self.scaler = joblib.load("models/lstm_scaler.save")
        self.lstm_model = LSTMModel(input_size=len(self.df.columns))
        self.lstm_model.load_state_dict(torch.load("models/lstm_safran.pth", map_location="cpu"))
        self.lstm_model.eval()

        self.initial_balance = initial_balance
        self.window_size = window_size
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * len(self.df.columns) + 3,), dtype=np.float32  # +1 pour LSTM, +2 balance/shares
        )
        self.reset()
        logging.info("🔁 Environnement réinitialisé")

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.window_size
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.previous_value = self.initial_balance
        return self._next_observation()

    def _next_observation(self):
        frame = self.df.iloc[self.current_step - self.window_size:self.current_step]
        obs = frame.to_numpy().flatten().tolist()
        obs.extend([self.balance, self.shares_held])

        lstm_input = self.df.iloc[self.current_step - SEQ_LEN:self.current_step].to_numpy()
        lstm_input = self.scaler.transform(lstm_input)
        lstm_input = torch.FloatTensor(lstm_input).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = self.lstm_model(lstm_input).item()
        obs.append(pred_scaled)

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']

        if action == 1 and self.balance > current_price:
            self.shares_held += 1
            self.balance -= current_price
            logging.info(f"🟢 BUY at {current_price:.2f}, Balance: {self.balance:.2f}, Shares: {self.shares_held}")
        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += current_price
            self.total_shares_sold += 1
            self.total_sales_value += current_price
            logging.info(f"🔴 SELL at {current_price:.2f}, Balance: {self.balance:.2f}, Shares: {self.shares_held}")

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        total_value = self.balance + self.shares_held * current_price
        reward = total_value - self.previous_value
        self.previous_value = total_value

        if done:
            profit = total_value - self.initial_balance
            logging.info(f"🏁 FINISHED | Total profit: {profit:.2f}")

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        profit = self.balance + self.shares_held * self.df.iloc[self.current_step]['close'] - self.initial_balance
        logging.info(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares held: {self.shares_held}, Profit: {profit:.2f}")
