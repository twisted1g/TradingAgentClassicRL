from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from gymnasium import spaces, Env


class MyTradingEnv(Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

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
        lambda_drawdown: float = 0.25,
        lambda_hold: float = 0.05,
        reward_scaling: float = 100.0,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        self.initial_balance = float(initial_balance)
        self.window_size = int(window_size)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.max_holding_time = int(max_holding_time)
        self.holding_threshold = int(holding_threshold)
        self.max_drawdown_threshold = float(max_drawdown_threshold)
        self.lambda_drawdown = float(lambda_drawdown)
        self.lambda_hold = float(lambda_hold)
        self.reward_scaling = float(reward_scaling)
        self.max_steps = int(max_steps) if max_steps is not None else None

        self.df = df.copy().reset_index(drop=True)
        self._prepare_data()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3, 2, 3])

        self.current_step = None
        self.position = 0
        self.units = 0.0
        self.entry_price = 0.0
        self.cash = 0.0
        self.portfolio_value = 0.0
        self.position_value = 0.0
        self.current_holding_time = 0
        self.max_drawdown = 0.0
        self.trade_history = []
        self._steps_elapsed = 0

    def _prepare_data(self):
        df = self.df
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        df["returns"] = df["close"].pct_change()
        df["rsi"] = self._calculate_rsi(df["close"])

        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_diff"] = df["macd"] - df["macd_signal"]

        df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
        df["price_to_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]
        df["price_trend"] = df["close"].diff(5).fillna(0)

        df["rsi_discrete"] = pd.cut(
            df["rsi"], bins=[-np.inf, 30, 70, np.inf], labels=[0, 1, 2]
        ).astype(pd.Int64Dtype())
        df["macd_discrete"] = pd.cut(
            df["macd_diff"], bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0, 1, 2]
        ).astype(pd.Int64Dtype())
        df["ma_trend_discrete"] = pd.cut(
            df["price_to_ma20"] * 100, bins=[-np.inf, -1, 1, np.inf], labels=[0, 1, 2]
        ).astype(pd.Int64Dtype())
        df["price_trend_discrete"] = pd.cut(
            df["price_trend"], bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2]
        ).astype(pd.Int64Dtype())

        df["rsi_discrete"] = df["rsi_discrete"].fillna(1)
        df["macd_discrete"] = df["macd_discrete"].fillna(1)
        df["ma_trend_discrete"] = df["ma_trend_discrete"].fillna(1)
        df["price_trend_discrete"] = df["price_trend_discrete"].fillna(1)

        df = df.bfill()
        df = df.ffill()
        df.fillna(0, inplace=True)

        self.df = df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        roll_up = up.rolling(window=period, min_periods=period).mean()
        roll_down = down.rolling(window=period, min_periods=period).mean()

        avg_gain = roll_up.copy()
        avg_loss = roll_down.copy()

        roll_up_i = roll_up.iloc[period - 1]
        roll_down_i = roll_down.iloc[period - 1]
        if np.isnan(roll_up_i):
            return pd.Series(np.nan, index=prices.index)

        avg_gain.iloc[period - 1] = roll_up_i
        avg_loss.iloc[period - 1] = roll_down_i

        for i in range(period, len(prices)):
            avg_gain.iloc[i] = (
                avg_gain.iloc[i - 1] * (period - 1) + up.iloc[i]
            ) / period
            avg_loss.iloc[i] = (
                avg_loss.iloc[i - 1] * (period - 1) + down.iloc[i]
            ) / period

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        rsi_level = int(row["rsi_discrete"])
        macd_signal = int(row["macd_discrete"])
        ma_trend = int(row["ma_trend_discrete"])
        price_trend = int(row["price_trend_discrete"])
        position_flag = int(self.position)

        if self.current_holding_time == 0:
            hold_level = 0
        elif self.current_holding_time < self.max_holding_time // 2:
            hold_level = 1
        else:
            hold_level = 2

        return np.array(
            [rsi_level, macd_signal, ma_trend, price_trend, position_flag, hold_level],
            dtype=np.int64,
        )

    def _calculate_reward(self, prev_portfolio_value: float) -> float:
        if prev_portfolio_value <= 0:
            return 0.0

        portfolio_pct_change = (
            (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value * 100.0
        )

        drawdown_penalty = 0.0
        hold_penalty = 0.0
        if self.position == 1 and self.current_holding_time > 0:
            drawdown_penalty = self.lambda_drawdown * (self.max_drawdown * 100.0)
            extra_hold = max(self.current_holding_time - self.holding_threshold, 0)
            hold_penalty = self.lambda_hold * np.log1p(extra_hold)

        reward = portfolio_pct_change - (drawdown_penalty + hold_penalty)
        return float(reward * self.reward_scaling)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)
        prev_portfolio_value = self.portfolio_value
        current_price = float(self.df.iloc[self.current_step]["close"])
        exit_reason = None

        if self.position == 0:
            if action == 1:
                price_with_slip = current_price * (1.0 + self.slippage)
                invest_amount = self.portfolio_value
                commission_fee = invest_amount * self.commission
                units = (invest_amount - commission_fee) / price_with_slip
                self.units = float(units)
                self.entry_price = price_with_slip
                self.position = 1
                self.current_holding_time = 0
                self.max_drawdown = 0.0
                self.cash = (
                    self.portfolio_value - units * price_with_slip - commission_fee
                )
                self.position_value = units * current_price
                self.portfolio_value = self.cash + self.position_value

        elif self.position == 1:
            self.current_holding_time += 1
            self.position_value = self.units * current_price
            self.portfolio_value = self.cash + self.position_value

            unrealized_pnl = self.position_value - (self.units * self.entry_price)
            current_drawdown = (
                -unrealized_pnl / (self.units * self.entry_price + 1e-12)
                if unrealized_pnl < 0
                else 0.0
            )
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

            should_close = (
                action == 2
                or self.current_holding_time >= self.max_holding_time
                or current_drawdown >= self.max_drawdown_threshold
            )
            if should_close:
                if action == 2:
                    exit_reason = "agent"
                elif self.current_holding_time >= self.max_holding_time:
                    exit_reason = "time"
                else:
                    exit_reason = "drawdown"

                exit_price = current_price * (1.0 - self.slippage)
                exit_value = self.units * exit_price
                commission_fee = exit_value * self.commission
                pnl = exit_value - (self.units * self.entry_price) - commission_fee

                self.cash += exit_value - commission_fee
                self.position_value = 0.0
                self.portfolio_value = self.cash

                self.trade_history.append(
                    {
                        "entry_price": self.entry_price,
                        "exit_price": exit_price,
                        "pnl": float(pnl),
                        "holding_time": int(self.current_holding_time),
                        "max_drawdown": float(self.max_drawdown),
                        "exit_reason": exit_reason,
                    }
                )

                self.units = 0.0
                self.entry_price = 0.0
                self.position = 0
                self.current_holding_time = 0
                self.max_drawdown = 0.0

        self.current_step += 1
        self._steps_elapsed += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = self.max_steps is not None and self._steps_elapsed >= self.max_steps
        reward = self._calculate_reward(prev_portfolio_value)
        obs = self._get_observation()

        info = {
            "portfolio_value": float(self.portfolio_value),
            "position": int(self.position),
            "holding_time": int(self.current_holding_time),
            "current_price": float(current_price),
            "n_trades": len(self.trade_history),
            "last_exit_reason": exit_reason,
        }

        return obs, reward, terminated, truncated, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        if self.max_steps is None:
            start_max = len(self.df) - 1
        else:
            start_max = len(self.df) - self.max_steps

        self.current_step = np.random.randint(self.window_size, start_max)
        self.position = 0
        self.units = 0.0
        self.entry_price = 0.0
        self.cash = float(self.initial_balance)
        self.portfolio_value = float(self.initial_balance)
        self.position_value = 0.0
        self.current_holding_time = 0
        self.max_drawdown = 0.0
        self.trade_history = []
        self._steps_elapsed = 0

        obs = self._get_observation()
        return obs, {}

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            step = self.current_step
            total = len(self.df) - 1
            print(
                f"Step: {step}/{total} | Portfolio: ${self.portfolio_value:,.2f} | "
                f"Position: {'Long' if self.position == 1 else 'Flat'} | "
                f"Hold time: {self.current_holding_time} | Trades: {len(self.trade_history)}"
            )

    def get_metrics(self) -> Dict[str, float]:
        if not self.trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_holding_time": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
            }

        trades = pd.DataFrame(self.trade_history)
        return {
            "total_trades": len(trades),
            "win_rate": float((trades["pnl"] > 0).mean() * 100.0),
            "avg_pnl": float(trades["pnl"].mean()),
            "avg_holding_time": float(trades["holding_time"].mean()),
            "max_drawdown": float(trades["max_drawdown"].max()),
            "total_pnl": float(trades["pnl"].sum()),
            "trades_closed_by_drawdown": int(
                (trades["exit_reason"] == "drawdown").sum()
            ),
            "trades_closed_by_time": int((trades["exit_reason"] == "time").sum()),
            "trades_closed_by_agent": int((trades["exit_reason"] == "agent").sum()),
        }
