from typing import Optional
import numpy as np
from .base_classical_agent import BaseClassicalAgent


class QLearningAgent(BaseClassicalAgent):
    def __init__(self, n_actions: int, **kwargs):
        super().__init__(n_actions=n_actions, name="Q-Learning", **kwargs)

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: Optional[int] = None,
    ):
        current_q = self.get_q_value(state, action)

        if done:
            target = reward
        else:
            next_state_key = self.state_to_key(next_state)
            max_next_q = np.max(self.q_table[next_state_key])
            target = reward + self.discount_factor * max_next_q

        new_q = current_q + self.learning_rate * (target - current_q)
        self.set_q_value(state, action, new_q)
