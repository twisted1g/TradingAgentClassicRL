import gymnasium as gym
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
        positions = [0, 1]

        def reward_function(self):
            pass

        def state_function(self):
            pass

        super().__init__(
            df=df,
            positions=positions,
            initial_position=0,
            trading_fees=commission,
            borrow_interest_rate=0,
            reward_function=reward_function,
            **kwargs,
        )

        self.max_holding_time = max_holding_time
        self.reward_scaling = reward_scaling
        self.slippage = slippage
        self.current_holding_time = 0
        self._prepare_data()

    def _prepare_data(self):
        self._add_technical_indicators()

    def _add_technical_indicators(self):
        df = self.df.copy()

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


df = pd.read_csv("./data/data_1h.csv")
trading_env = MyTradingEnv(df=df)
print(trading_env.df)
