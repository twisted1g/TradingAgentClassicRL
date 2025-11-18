from typing import Dict, Any, Optional
import numpy as np
from .base_agent import BaseAgent

RANDOM_SEED = 42


class RandomAgent(BaseAgent):
    def __init__(
        self,
        name="RandomAgent",
        seed: Optional[int] = RANDOM_SEED,
        buy_prob=0.1,
        sell_prob=0.1,
        hold_prob=0.8,
    ):
        super().__init__(name=name, seed=seed)

        total = buy_prob + hold_prob
        self.buy_prob = buy_prob / total
        self.hold_prob = hold_prob / total
        self.close_prob = sell_prob

    def act(
        self,
        observation: np.ndarray,
        info: Dict[str, Any],
    ) -> int:
        if "position" in info:
            self.position = info["position"]
        elif len(observation) >= 5:
            self.position = observation[4]

        if self.position == 0:
            action = np.random.choice(
                [0, 1],
                p=[self.hold_prob, self.buy_prob],
            )
        else:
            action = np.random.choice(
                [0, 2],
                p=[1 - self.close_prob, self.close_prob],
            )

        return action

    def __str__(self) -> str:
        return f"{self.name} (Buy={self.buy_prob:.2f}, Close={self.close_prob:.2f})"
