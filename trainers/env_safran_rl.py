import gym
import numpy as np
from gym import spaces

class SafranTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(SafranTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.df.columns) + 2,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.iloc[self.current_step].values.tolist()
        obs.extend([self.balance, self.shares_held])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0

        if action == 1 and self.balance > current_price:  # Buy
            self.shares_held += 1
            self.balance -= current_price
        elif action == 2 and self.shares_held > 0:  # Sell
            self.shares_held -= 1
            self.balance += current_price
            self.total_shares_sold += 1
            self.total_sales_value += current_price

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward = self.balance + self.shares_held * current_price - self.initial_balance if done else 0
        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        profit = self.balance + self.shares_held * self.df.iloc[self.current_step]['close'] - self.initial_balance
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares held: {self.shares_held}, Profit: {profit:.2f}")