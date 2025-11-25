from typing import Dict, Any
import numpy as np
from .base_agent import BaseAgent


class BuyAndHoldAgent(BaseAgent):
    def __init__(
        self,
        name: str = "BuyAndHoldAgent",
        initial_buy_step: int = 0,
    ):
        super().__init__(name=name, seed=None)
        self.initial_buy_step = initial_buy_step
        self.has_bought = False

    def act(
        self,
        observation: np.ndarray,
        info: Dict[str, Any],
    ) -> int:
        if "position" in info:
            self.position = info["position"]

        if not self.has_bought and self.current_step >= self.initial_buy_step:
            if self.position == 0:
                self.has_bought = True
                return 1

        return 0

    def reset(self):
        super().reset()
        self.has_bought = False

    def __str__(self) -> str:
        status = "Bought" if self.has_bought else "Waiting"
        return f"{self.name} ({status}, Step: {self.current_step})"
