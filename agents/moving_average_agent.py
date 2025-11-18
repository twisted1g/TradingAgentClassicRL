from typing import Dict, Any
import numpy as np
from .base_agent import BaseAgent


class MovingAverageAgent(BaseAgent):
    def __init__(
        self,
        name: str = "MovingAverageAgent",
        fast_period: int = 10,
        slow_period: int = 30,
        use_exponential: bool = False,
    ):

        super().__init__(name=name, seed=None)

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_exponential = use_exponential

        self.price_buffer = []
        self.fast_ma = None
        self.slow_ma = None
        self.prev_fast_ma = None
        self.prev_slow_ma = None

    def _calculate_ma(self, prices: list, period: int) -> float:
        if len(prices) < period:
            return None

        if self.use_exponential:
            alpha = 2 / (period + 1)
            ema = prices[-period]
            for price in prices[-period + 1 :]:
                ema = alpha * price + (1 - alpha) * ema
            return ema
        else:
            return np.mean(prices[-period:])

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> int:

        current_price = info.get("current_price", 0)
        if current_price == 0 and len(observation) > 0:
            current_price = 100.0

        self.price_buffer.append(current_price)

        if "position" in info:
            self.position = info["position"]

        self.prev_fast_ma = self.fast_ma
        self.prev_slow_ma = self.slow_ma

        self.fast_ma = self._calculate_ma(self.price_buffer, self.fast_period)
        self.slow_ma = self._calculate_ma(self.price_buffer, self.slow_period)

        if self.fast_ma is None or self.slow_ma is None:
            return 0

        if self.prev_fast_ma is None or self.prev_slow_ma is None:
            return 0

        action = 0

        if self.prev_fast_ma <= self.prev_slow_ma and self.fast_ma > self.slow_ma:
            if self.position == 0:
                action = 1

        elif self.prev_fast_ma >= self.prev_slow_ma and self.fast_ma < self.slow_ma:
            if self.position == 1:
                action = 2

        elif self.position == 1:
            if self.fast_ma < self.slow_ma * 0.98:
                action = 2

        return action

    def reset(self):
        super().reset()
        self.price_buffer = []
        self.fast_ma = None
        self.slow_ma = None
        self.prev_fast_ma = None
        self.prev_slow_ma = None

    def get_current_mas(self) -> Dict[str, float]:
        return {
            "fast_ma": self.fast_ma,
            "slow_ma": self.slow_ma,
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
        }

    def __str__(self) -> str:
        ma_type = "EMA" if self.use_exponential else "SMA"
        return (
            f"{self.name} ({ma_type} {self.fast_period}/{self.slow_period}, "
            f"Fast={self.fast_ma:.2f if self.fast_ma else 0:.2f}, "
            f"Slow={self.slow_ma:.2f if self.slow_ma else 0:.2f})"
        )
