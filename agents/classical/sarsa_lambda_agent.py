from typing import Dict, Optional
import numpy as np
import pickle
from collections import defaultdict
from .base_classical_agent import BaseClassicalAgent


class SarsaLambdaAgent(BaseClassicalAgent):
    def __init__(
        self,
        lambda_param: float = 0.9,
        replace_traces: bool = True,
        name: str = "SARSA(lambda)",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.lambda_param = lambda_param
        self.replace_traces = replace_traces

        self.eligibility_traces = defaultdict(lambda: np.zeros(self.n_actions))

    def reset_eligibility_traces(self):
        self.eligibility_traces.clear()

    def update_eligibility_traces(self, state: np.ndarray, action: int):
        state_key = self.state_to_key(state)

        if self.replace_traces:
            self.eligibility_traces[state_key] = np.zeros(self.n_actions)
            self.eligibility_traces[state_key][action] = 1.0
        else:
            self.eligibility_traces[state_key][action] += 1.0

    def decay_eligibility_traces(self):
        decay_factor = self.lambda_param * self.discount_factor

        for state_key in list(self.eligibility_traces.keys()):
            self.eligibility_traces[state_key] *= decay_factor

            if np.max(np.abs(self.eligibility_traces[state_key])) < 1e-6:
                del self.eligibility_traces[state_key]

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: Optional[int] = None,
    ):
        if next_action is None:
            next_action = self.select_action(next_state, training=True)

        current_q = self.get_q_value(state, action)

        if done:
            target = reward
        else:
            next_q = self.get_q_value(next_state, next_action)
            target = reward + self.discount_factor * next_q

        td_error = target - current_q

        self.update_eligibility_traces(state, action)

        for state_key, traces in self.eligibility_traces.items():
            state_array = np.array(state_key)
            for action_idx in range(self.n_actions):
                if traces[action_idx] != 0:
                    current_q_val = self.get_q_value(state_array, action_idx)

                    new_q = (
                        current_q_val
                        + self.learning_rate * td_error * traces[action_idx]
                    )
                    self.set_q_value(state_array, action_idx, new_q)

        self.decay_eligibility_traces()

        if done:
            self.reset_eligibility_traces()

    def train_episode(
        self, env, max_steps: int = 1000, verbose: bool = False
    ) -> Dict[str, float]:
        self.reset_eligibility_traces()

        state, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        action = self.select_action(state, training=True)

        while not done and steps < max_steps:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_action = self.select_action(next_state, training=True)

            self.update(state, action, reward, next_state, done, next_action)

            state = next_state
            action = next_action
            episode_reward += reward
            steps += 1
            self.total_steps += 1

        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lens.append(steps)
        self.decay_epsilon()

        if len(self.q_table) > 0:
            avg_q = np.mean([np.max(q_vals) for q_vals in self.q_table.values()])
            self.q_value_history.append(avg_q)

        if verbose and self.episode_count % 100 == 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            print(
                f"Episode {self.episode_count}: "
                f"Reward={episode_reward:.2f}, "
                f"Avg100={avg_reward:.2f}, "
                f"Epsilon={self.epsilon:.3f}, "
                f"Q-table size={len(self.q_table)}, "
                f"Traces size={len(self.eligibility_traces)}"
            )

        return {
            "episode": self.episode_count,
            "reward": episode_reward,
            "steps": steps,
            "epsilon": self.epsilon,
        }

    def save(self, filepath: str):
        data = {
            "name": self.name,
            "q_table": dict(self.q_table),
            "n_actions": self.n_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "episode_rewards": self.episode_rewards,
            "lambda_param": self.lambda_param,
            "replace_traces": self.replace_traces,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"SARSA(λ) агент сохранён: {filepath}")

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.name = data["name"]
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions),
            {k: np.array(v) for k, v in data["q_table"].items()},
        )
        self.learning_rate = data["learning_rate"]
        self.discount_factor = data["discount_factor"]
        self.epsilon = data["epsilon"]
        self.episode_count = data["episode_count"]
        self.episode_rewards = data["episode_rewards"]
        self.lambda_param = data.get("lambda_param", 0.9)
        self.replace_traces = data.get("replace_traces", True)

        self.reset_eligibility_traces()

        print(f"SARSA(λ) агент загружен: {filepath}")
