from typing import Dict, Any
import numpy as np
from .base_agent import BaseAgent


class MovingAverageAgent(BaseAgent):
    def __init__(
        self,
        name: str = "MovingAverageAgent",
        fast_period: int = 20,
        slow_period: int = 50,
        use_exponential: bool = False,
    ):
        super().__init__(name=name, seed=None)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_exponential = use_exponential

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> int:

        if len(observation) < 6:
            return 0

        ma_trend = int(observation[2])
        position = int(observation[4])

        self.position = position

        if ma_trend == 2 and position == 0:
            return 1
        elif ma_trend == 0 and position == 1:
            return 2
        else:
            return 0

    def reset(self):
        super().reset()

    def get_current_mas(self) -> Dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "use_exponential": self.use_exponential,
        }

    def __str__(self) -> str:
        return (
            f"{self.name} (using ma_trend_discrete from env, "
            f"fast={self.fast_period}, slow={self.slow_period})"
        )
