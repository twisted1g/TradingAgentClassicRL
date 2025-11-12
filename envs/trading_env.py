from typing import Any, Dict, Tuple
import gymnasium as gym
from gymnasium import spaces
from gym_trading_env.environments import TradingEnv
import pandas as pd
import numpy as np


class MyTradingEnv(TradingEnv):

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1000.0,
        window_size: int = 10,
        commission: float = 0.0001,
        slippage: float = 0.0005,
        max_holding_time: int = 60 * 24,
        lambda_drawdown: float = 0.5,  # в будущем продобрать гиперпараметры
        lambda_hold: float = 0.1,  # в будущем продобрать гиперпараметры
        reward_scaling=1.0,
        **kwargs,
    ):
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.commission = commission
        self.slippage = slippage
        self.max_holding_time = max_holding_time
        self.reward_scaling = reward_scaling
        self.lambda_drawdown = lambda_drawdown
        self.lambda_hold = lambda_hold
        self.reward_scaling = reward_scaling

        positions = [0, 1]

        self.df = df.copy()
        self._prepare_data()

        super().__init__(
            df=self.df,
            positions=positions,
            initial_position=0,
            trading_fees=commission,
            borrow_interest_rate=0,
            portfolio_initial_value=initial_balance,
            **kwargs,
        )

        self.current_holding_time = 0
        self.entry_price = 0.0
        self.max_drawdown = 0
        self.trade_history = []

        self.observation_space = spaces.MultiDiscrete([3, 2, 3])  # добавить индикаторы
        self.action_space = spaces.Discrete(3)

    def _prepare_data(self):
        self._add_technical_indicators()
        self._discretize_features()

        self.df.fillna(method="bfill", inplace=True)
        self.df.fillna(method="ffill", inplace=True)

    def _add_technical_indicators(self):  # добавить индикаторы
        df = self.df

        df["returns"] = df["close"].pct_change()

        df["rsi"] = self._calculate_rsi(df["close"])

        self.df = df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:

        delta: pd.Series = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss

        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _discretize_features(self):  # добавить индикаторы
        df = self.df

        df["rsi_discrete"] = pd.cut(
            df["rsi"],
            bins=[-np.inf, 30, 70, np.inf],
            labels=[0, 1, 2],
        ).astype(int)

    def _get_observation(self) -> np.ndarray:  # добавить индикаторы
        if self._idx >= len(self.df):
            return np.array([1, 0, 0])

        row = self.df.iloc[self._idx]

        rsi_level = int(row["rsi_discrete"])

        position = self._position

        if self.current_holding_time == 0:
            hold_level = 0
        elif self.current_holding_time < self.max_holding_time // 2:
            hold_level = 1
        else:
            hold_level = 2

        return np.array([rsi_level, position, hold_level])

    def _calculate_reward(self, action: int, perv_portfolio_value: float) -> float:
        current_price = self.df.iloc[self._idx]["close"]

        portfolio_change = self._portfolio_value - perv_portfolio_value

        if self._position == 1 and self.current_holding_time > 0:
            unrealized_pnl = current_price - self.entry_price

            if unrealized_pnl < 0:
                self.max_drawdown = max(self.max_drawdown, abs(unrealized_pnl))

        drawdown_penalty = self.lambda_drawdown * self.max_drawdown
        hold_penalty = self.lambda_hold * self.current_holding_time

        reward = float(portfolio_change - (drawdown_penalty + hold_penalty))

        if action == 2 and self.current_holding_time > 0:
            pnl = current_price - self.entry_price
            self.trade_history.append(
                {
                    "entry_price": self.entry_price,
                    "exit_price": current_price,
                    "pnl": pnl,
                    "holding_time": self.current_holding_time,
                    "max_drawdown": self.max_drawdown,
                }
            )

        return reward * self.reward_scaling

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, Dict[str, Any]]:
        perv_portfolio_value = self._portfolio_value
        current_price = self.df.iloc[self._idx]["close"]

        actual_action = action

        if self._position == 0:
            if action == 1:
                actual_action = 1
                self.entry_price = current_price * (1 + self.slippage)
                self.current_holding_time = 0
                self.max_drawdown = 0.0
            else:
                actual_action = 0

        elif self._position == 1:
            self.current_holding_time += 1

            if action == 2:
                actual_action = 0

            elif self.current_holding_time >= self.max_holding_time:
                actual_action = 0

            else:
                actual_action = 1

        self._position = actual_action
        self._idx += 1

        terminated = self._idx >= len(self.df) - 1
        truncated = False

        reward = self._calculate_reward(
            action=action, perv_portfolio_value=perv_portfolio_value
        )

        if self._position == 1:
            position_value = self.initial_balance * (current_price / self.entry_price)
            self._portfolio_value = position_value
        else:
            self._portfolio_value = self.initial_balance

            if self.current_holding_time > 0:
                
        

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed, options=options)

        self._idx = self.window_size
        self._position = 0
        self.current_holding_time = 0
        self.entry_price = 0.0
        self.max_drawdown = 0.0
        self._portfolio_value = self.initial_balance
        self.trade_history = []

        observation = self._get_observation()
        info = {}

        return observation, info

    def render(self):
        pass

    def get_metrics(self):
        return super().get_metrics()


df = pd.read_csv("./data/data_1h.csv")
trading_env = MyTradingEnv(df=df)
print(trading_env.df)
