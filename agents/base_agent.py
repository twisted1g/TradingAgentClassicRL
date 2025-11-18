from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseAgent(ABC):
    def __init__(
        self,
        name: str = "BaseAgent",
        initial_balance: int = 1000,
        seed: Optional[int] = None,
    ):
        self.name = name
        self.initial_balance = initial_balance
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.reset()

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,
        info: Dict[str, Any],
    ) -> int:
        pass

    def reset(self):
        self.position = 0
        self.entry_price = 0.0
        self.current_step = 0
        self.history = {
            "actions": [],
            "positions": [],
            "prices": [],
            "rewards": [],
        }

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: Dict[str, Any],
    ):
        self.history["actions"].append(action)
        self.history["positions"].append(self.position)
        self.history["rewards"].append(reward)

        if "current_price" in info:
            self.history["prices"].append(info["current_price"])

        self.current_step += 1

    def get_status(self) -> Dict[str, Any]:
        actions = np.array(self.history["actions"])
        rewards = np.array(self.history["rewards"])

        stats = {
            "agent_name": self.name,
            "total_steps": len(actions),
            "total_reward": np.sum(rewards),
            "mean_reward": np.mean(rewards) if len(rewards) > 0 else 0,
            "num_buys": np.sum(actions == 1),
            "num_sells": np.sum(actions == 2),
            "num_holds": np.sum(actions == 0),
        }

        return stats

    def __str__(self) -> str:
        return f"{self.name} Position: {self.position}, Step: {self.current_step}"

    def __repr__(self) -> str:
        return self.__str__()
