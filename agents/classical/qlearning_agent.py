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
        state_key = self.state_to_key(state)
        current_q = self.q_table[state_key][action]

        if done:
            target = reward
        else:
            next_state_key = self.state_to_key(next_state)
            max_next_q = np.max(self.q_table[next_state_key])
            target = reward + self.discount_factor * max_next_q

        td_error = target - current_q
        self.q_table[state_key][action] += self.learning_rate * td_error

        self.td_error_history.append(td_error)
