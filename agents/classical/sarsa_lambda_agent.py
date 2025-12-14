from typing import Dict, Optional
import numpy as np
from collections import defaultdict
from .base_classical_agent import BaseClassicalAgent


class SarsaLambdaAgent(BaseClassicalAgent):

    def __init__(
        self,
        lambda_param: float = 0.6,
        replace_traces: bool = True,
        name: str = "SARSA(lambda)",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.lambda_param = lambda_param
        self.replace_traces = replace_traces

        self.eligibility_traces = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )

    def reset_eligibility_traces(self):
        self.eligibility_traces.clear()

    def update_eligibility_traces(self, state: np.ndarray, action: int):
        state_key = self.state_to_key(state)

        if self.replace_traces:
            self.eligibility_traces[state_key] = np.zeros(
                self.n_actions, dtype=np.float32
            )
            self.eligibility_traces[state_key][action] = 1.0
        else:
            self.eligibility_traces[state_key][action] += 1.0

    def decay_eligibility_traces(self):
        decay_factor = self.discount_factor * self.lambda_param
        keys_to_delete = []

        for state_key, traces in self.eligibility_traces.items():
            traces *= decay_factor
            if np.max(np.abs(traces)) < 1e-6:
                keys_to_delete.append(state_key)

        for key in keys_to_delete:
            del self.eligibility_traces[key]

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: Optional[int] = None,
    ):
        if not done and next_action is None:
            raise ValueError("SARSA requires next_action for non-terminal states")

        state_key = self.state_to_key(state)
        current_q = self.q_table[state_key][action]

        if done:
            target = reward
        else:
            next_state_key = self.state_to_key(next_state)
            target = (
                reward
                + self.discount_factor * self.q_table[next_state_key][next_action]
            )

        td_error = target - current_q

        self.update_eligibility_traces(state, action)

        for trace_state_key, traces in self.eligibility_traces.items():
            for a in range(self.n_actions):
                if traces[a] > 1e-6:
                    self.q_table[trace_state_key][a] += (
                        self.learning_rate * td_error * traces[a]
                    )

        self.decay_eligibility_traces()

        self.td_error_history.append(td_error)

        if done:
            self.reset_eligibility_traces()

    def train_episode(
        self, env, max_steps: int = 1000, verbose: bool = False
    ) -> Dict[str, float]:
        self.reset_eligibility_traces()

        state, info = env.reset()
        action = self.select_action(state, training=True)

        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if not done:
                next_action = self.select_action(next_state, training=True)
            else:
                next_action = None

            self.update(state, action, reward, next_state, done, next_action)

            episode_reward += reward
            steps += 1
            self.total_steps += 1

            if not done:
                state = next_state
                action = next_action

        self.episode_count += 1
        self.episode_rewards.append(float(episode_reward))
        self.episode_lens.append(int(steps))
        self.decay_epsilon()

        if verbose and self.episode_count % 100 == 0:
            print(
                f"Episode {self.episode_count} | "
                f"Reward={episode_reward:.2f} | "
                f"Epsilon={self.epsilon:.3f} | "
                f"Traces={len(self.eligibility_traces)}"
            )

        return {
            "episode": self.episode_count,
            "reward": float(episode_reward),
            "steps": int(steps),
            "epsilon": float(self.epsilon),
            "portfolio_value": float(info.get("portfolio_value", env.initial_balance)),
            "n_trades": int(info.get("n_trades", 0)),
        }
