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
        decay = self.discount_factor * self.lambda_param
        to_delete = []

        for state_key, traces in self.eligibility_traces.items():
            traces *= decay
            if np.max(np.abs(traces)) < 1e-6:
                to_delete.append(state_key)

        for k in to_delete:
            del self.eligibility_traces[k]

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: Optional[int] = None,
    ):
        # SARSA — строго on-policy
        if not done and next_action is None:
            raise ValueError("SARSA requires next_action for non-terminal states")

        current_q = self.get_q_value(state, action)

        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.get_q_value(
                next_state, next_action
            )

        td_error = target - current_q

        # 1️⃣ обновляем eligibility trace
        self.update_eligibility_traces(state, action)

        # 2️⃣ обновляем все Q по traces
        for state_key, traces in self.eligibility_traces.items():
            for a in range(self.n_actions):
                if traces[a] == 0.0:
                    continue

                q = self.q_table[state_key][a]
                self.q_table[state_key][a] = q + self.learning_rate * td_error * traces[a]

        # 3️⃣ decay traces
        self.decay_eligibility_traces()

        # 4️⃣ адаптивный LR
        self.update_adaptive_learning_rate(td_error)

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

            state = next_state
            action = next_action

        self.episode_count += 1
        self.episode_rewards.append(float(episode_reward))
        self.episode_lens.append(int(steps))
        self.decay_epsilon()

        if self.q_table:
            avg_q = np.mean([np.max(v) for v in self.q_table.values()])
            self.q_value_history.append(float(avg_q))

        if verbose and self.episode_count % 100 == 0:
            print(
                f"Episode {self.episode_count} | "
                f"Reward={episode_reward:.2f} | "
                f"Epsilon={self.epsilon:.3f} | "
                f"LR={self.learning_rate:.4f} | "
                f"Traces={len(self.eligibility_traces)}"
            )

        return {
            "episode": self.episode_count,
            "reward": float(episode_reward),
            "steps": int(steps),
            "epsilon": float(self.epsilon),
        }
