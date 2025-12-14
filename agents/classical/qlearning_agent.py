from typing import Optional
import numpy as np
from .base_classical_agent import BaseClassicalAgent


class QLearningAgent(BaseClassicalAgent):
    def __init__(self, **kwargs):
        super().__init__(name="Q-Learning", **kwargs)

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
            max_next_q = np.max(self.q_table[self.state_to_key(next_state)])
            target = reward + self.discount_factor * max_next_q

        td_error = target - current_q
        new_q = current_q + self.learning_rate * td_error
        self.set_q_value(state, action, new_q)

        self.update_adaptive_learning_rate(td_error)