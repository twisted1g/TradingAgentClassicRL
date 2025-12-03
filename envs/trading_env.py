from typing import Any, Dict, Tuple
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
        holding_threshold: int = 24,
        max_drawdown_threshold: float = 0.05,
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
        self.max_drawdown_threshold = max_drawdown_threshold
        self.holding_threshold = holding_threshold
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
        self.max_drawdown = 0.0
        self.position_value = 0.0
        self.cash_after_entry = 0.0
        self.trade_history = []

        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3, 2, 3])
        self.action_space = spaces.Discrete(3)

    def _prepare_data(self):
        self._add_technical_indicators()
        self._discretize_features()

        self.df.bfill(inplace=True)
        self.df.ffill(inplace=True)
        self.df.fillna(0, inplace=True)

    def _add_technical_indicators(self):
        df = self.df

        df["returns"] = df["close"].pct_change()

        df["rsi"] = self._calculate_rsi(df["close"])

        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_diff"] = df["macd"] - df["macd_signal"]

        df["ma_20"] = df["close"].rolling(window=20).mean()
        df["ma_50"] = df["close"].rolling(window=50).mean()

        df["price_to_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]

        df["price_trend"] = df["close"].diff(5)

        self.df = df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:

        delta: pd.Series = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss

        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _discretize_features(self):
        df = self.df

        df["rsi_discrete"] = pd.cut(
            df["rsi"],
            bins=[-np.inf, 30, 70, np.inf],
            labels=[0, 1, 2],
        )
        df["rsi_discrete"] = df["rsi_discrete"].fillna(1).astype(int)

        df["macd_discrete"] = pd.cut(
            df["macd_diff"],
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=[0, 1, 2],
        )
        df["macd_discrete"] = df["macd_discrete"].fillna(1).astype(int)

        df["ma_trend_discrete"] = pd.cut(
            df["price_to_ma20"] * 100,
            bins=[-np.inf, -1, 1, np.inf],
            labels=[0, 1, 2],
        )
        df["ma_trend_discrete"] = df["ma_trend_discrete"].fillna(1).astype(int)

        df["price_trend_discrete"] = pd.cut(
            df["price_trend"],
            bins=[-np.inf, -0.01, 0.01, np.inf],
            labels=[0, 1, 2],
        )
        df["price_trend_discrete"] = df["price_trend_discrete"].fillna(1).astype(int)

    def _get_observation(self) -> np.ndarray:
        if self._idx >= len(self.df):
            return np.array([1, 0, 0])

        row = self.df.iloc[self._idx]

        rsi_level = int(row["rsi_discrete"])
        macd_signal = int(row["macd_discrete"])
        ma_trend = int(row["ma_trend_discrete"])
        price_trend = int(row["price_trend_discrete"])

        position = self._position

        if self.current_holding_time == 0:
            hold_level = 0
        elif self.current_holding_time < self.max_holding_time // 2:
            hold_level = 1
        else:
            hold_level = 2

        return np.array(
            [
                rsi_level,
                macd_signal,
                ma_trend,
                price_trend,
                position,
                hold_level,
            ]
        )

    def _calculate_reward(self, prev_portfolio_value: float) -> float:
        portfolio_change = self._portfolio_value - prev_portfolio_value

        drawdown_penalty = 0.0
        hold_penalty = 0.0

        if self._position == 1 and self.current_holding_time > 0:
            drawdown_penalty = self.lambda_drawdown * self.max_drawdown
            hold_penalty = self.lambda_hold * max(
                self.current_holding_time - self.holding_threshold, 0
            )

        reward = float(portfolio_change - (drawdown_penalty + hold_penalty))

        return reward * self.reward_scaling

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, Dict[str, Any]]:
        prev_portfolio_value = self._portfolio_value
        current_price = self.df.iloc[self._idx]["close"]

        actual_action = action

        if self._position == 0:
            if action == 1:
                entry_price_with_slippage = current_price * (1 + self.slippage)
                invest_amount = self._portfolio_value
                entry_commission = invest_amount * self.commission

                self.entry_price = entry_price_with_slippage
                self.position_value = invest_amount
                self.cash_after_entry = -entry_commission
                self._portfolio_value -= entry_commission

                self.current_holding_time = 0
                self.max_drawdown = 0.0
                actual_action = 1
            else:
                actual_action = 0

        elif self._position == 1:
            self.current_holding_time += 1

            current_position_value = self.position_value * (
                current_price / self.entry_price
            )
            self._portfolio_value = self.cash_after_entry + current_position_value

            unrealized_pnl = current_position_value - self.position_value
            current_drawdown = max(0.0, -unrealized_pnl / self.position_value)
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

            should_close = False
            exit_reason = None

            if action == 2:
                should_close = True
                exit_reason = "agent"
            elif self.current_holding_time >= self.max_holding_time:
                should_close = True
                exit_reason = "time"
            elif current_drawdown >= self.max_drawdown_threshold:
                should_close = True
                exit_reason = "drawdown"

            if should_close:
                exit_price_with_slippage = current_price * (1 - self.slippage)
                units = self.position_value / self.entry_price
                exit_value = units * exit_price_with_slippage
                exit_commission = exit_value * self.commission
                pnl = exit_value - self.position_value - exit_commission

                self._portfolio_value = (
                    self.cash_after_entry + exit_value - exit_commission
                )

                self.trade_history.append(
                    {
                        "entry_price": self.entry_price,
                        "exit_price": exit_price_with_slippage,
                        "pnl": pnl,
                        "holding_time": self.current_holding_time,
                        "max_drawdown": self.max_drawdown,
                        "exit_reason": exit_reason,
                    }
                )

                self.entry_price = 0.0
                self.position_value = 0.0
                self.cash_after_entry = 0.0
                self.current_holding_time = 0
                self.max_drawdown = 0.0
                actual_action = 0
            else:
                actual_action = 1

        self._position = actual_action
        self._idx += 1

        terminated = self._idx >= len(self.df) - 1
        truncated = False

        reward = self._calculate_reward(prev_portfolio_value=prev_portfolio_value)

        observation = self._get_observation()

        info = {
            "portfolio_value": self._portfolio_value,
            "position": self._position,
            "holding_time": self.current_holding_time,
            "current_price": current_price,
            "n_trades": len(self.trade_history),
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed, options=options)

        self._idx = self.window_size
        self._position = 0
        self.current_holding_time = 0
        self.entry_price = 0.0
        self.max_drawdown = 0.0
        self._portfolio_value = self.initial_balance
        self.position_value = 0.0
        self.cash_after_entry = 0.0
        self.trade_history = []

        observation = self._get_observation()
        info = {}

        return observation, info

    def render(self, mode="human"):
        if mode == "human":
            print(
                f"Step: {self._idx} / {len(self.df)-1},"
                f"Portfolio: ${self._portfolio_value:.2f},"
                f"Position: {'Long' if self._position == 1 else 'Flat'},"
                f"Hold Time: {self.current_holding_time}, "
                f"Trades: {len(self.trade_history)}"
            )

    def get_metrics(self) -> Dict[str, float]:
        if len(self.trade_history) == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_holding_time": 0.0,
            }

        trades = pd.DataFrame(self.trade_history)

        return {
            "total_trades": len(trades),
            "win_rate": (trades["pnl"] > 0).mean() * 100,
            "avg_pnl": trades["pnl"].mean(),
            "avg_holding_time": trades["holding_time"].mean(),
            "max_drawdown": trades["max_drawdown"].max(),
            "total_pnl": trades["pnl"].sum(),
            "trades_closed_by_drawdown": (trades["exit_reason"] == "drawdown").sum(),
            "trades_closed_by_time": (trades["exit_reason"] == "time").sum(),
        }
