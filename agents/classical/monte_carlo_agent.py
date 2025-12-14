from typing import Dict, Optional, List, Tuple, Set
import numpy as np
from collections import defaultdict
from .base_classical_agent import BaseClassicalAgent


class MonteCarloAgent(BaseClassicalAgent):
    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: Optional[float] = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        name: str = "MonteCarlo",
        first_visit: bool = True,
        use_sample_average: bool = False,
        return_clip: Optional[Tuple[float, float]] = None,
    ):
        super().__init__(
            n_actions=n_actions,
            learning_rate=learning_rate if learning_rate is not None else 0.1,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            name=name,
        )

        self.first_visit = first_visit
        self.use_sample_average = use_sample_average
        self.return_clip = return_clip

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.returns_count = defaultdict(int)

        self.episode_transitions: List[Tuple[tuple, int, float]] = []

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
        self.episode_transitions.append((state_key, action, float(reward)))

        if not done:
            return

        G = 0.0
        visited: Set[Tuple] = set()

        for t in reversed(range(len(self.episode_transitions))):
            s, a, r = self.episode_transitions[t]
            sa = (s, a)

            G = r + self.discount_factor * G

            if self.return_clip is not None:
                low, high = self.return_clip
                G = float(np.clip(G, low, high))

            if self.first_visit and sa in visited:
                continue

            visited.add(sa)

            current_q = self.q_table[s][a]

            if self.use_sample_average:
                self.returns_count[sa] += 1
                alpha = 1.0 / self.returns_count[sa]
            else:
                alpha = self.learning_rate

            self.q_table[s][a] = current_q + alpha * (G - current_q)

        self.episode_transitions.clear()

    def train_episode(
        self, env, max_steps: int = 1000, verbose: bool = False
    ) -> Dict[str, float]:

        state, info = env.reset()
        done = False
        steps = 0
        episode_reward = 0.0

        action = self.select_action(state, training=True)

        while not done and steps < max_steps:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.update(state, action, reward, next_state, done)

            episode_reward += reward
            steps += 1
            self.total_steps += 1

            if done:
                break

            state = next_state
            action = self.select_action(state, training=True)

        self.episode_count += 1
        self.episode_rewards.append(float(episode_reward))
        self.episode_lens.append(int(steps))
        self.decay_epsilon()

        if self.q_table:
            avg_q = np.mean([np.max(v) for v in self.q_table.values()])
            self.q_value_history.append(float(avg_q))

        return {
            "episode": self.episode_count,
            "reward": float(episode_reward),
            "steps": int(steps),
            "epsilon": float(self.epsilon),
            "portfolio_value": float(info.get("portfolio_value", 0)),
            "n_trades": int(info.get("n_trades", 0)),
        }
