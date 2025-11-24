import numpy as np
import pickle
from .base_classical_agent import BaseClassicalAgent
from collections import defaultdict
from typing import Dict


class CrossEntropyAgent(BaseClassicalAgent):
    def __init__(
        self,
        n_actions: int,
        elite_percentile: float = 0.1,
        min_samples: int = 100,
        name: str = "CrossEntropyAgent",
    ):
        super().__init__(
            n_actions=n_actions,
            learning_rate=0.0,
            discount_factor=1.0,
            epsilon_start=0.0,
            name=name,
        )
        self.elite_percentile = elite_percentile
        self.min_samples = min_samples

        self.policy = defaultdict(lambda: np.ones(n_actions) / n_actions)

        self.epsilon = 0.0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state_key = self.state_to_key(state)
        probs = self.policy[state_key]
        return np.random.choice(self.n_actions, p=probs)

    def update(self, *args, **kwargs):
        pass

    def train(
        self, env, n_episodes: int, max_steps: int = 1000, verbose: bool = True
    ) -> Dict[str, list]:
        print(f"Начало обучения {self.name} (Cross-Entropy Method)")
        print(
            f"Elite percentile: {self.elite_percentile}, Min samples: {self.min_samples}"
        )
        print("=" * 70)

        batch_size = max(n_episodes, self.min_samples)

        for iteration in range(100):
            trajectories = []
            total_rewards = []

            for _ in range(batch_size):
                state, _ = env.reset()
                done = False
                episode_reward = 0
                steps = 0
                trajectory = []

                while not done and steps < max_steps:
                    action = self.select_action(state, training=True)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    trajectory.append((self.state_to_key(state), action))
                    episode_reward += reward
                    state = next_state
                    steps += 1

                trajectories.append(trajectory)
                total_rewards.append(episode_reward)

            threshold = np.percentile(total_rewards, (1 - self.elite_percentile) * 100)
            elite_trajectories = [
                traj for traj, r in zip(trajectories, total_rewards) if r >= threshold
            ]

            if not elite_trajectories:
                if verbose:
                    print("⚠️ Нет элитных траекторий — пропуск обновления.")
                continue

            action_counts = defaultdict(lambda: np.zeros(self.n_actions))
            state_counts = defaultdict(int)

            for traj in elite_trajectories:
                for state_key, action in traj:
                    action_counts[state_key][action] += 1
                    state_counts[state_key] += 1

            for state_key in action_counts:
                self.policy[state_key] = (
                    action_counts[state_key] / state_counts[state_key]
                )

            mean_reward = np.mean(total_rewards)
            elite_mean = np.mean([r for r in total_rewards if r >= threshold])
            if verbose:
                print(
                    f"Iter {iteration+1}: "
                    f"Mean={mean_reward:.2f}, "
                    f"Elite={elite_mean:.2f}, "
                    f"Threshold={threshold:.2f}, "
                    f"Policy states={len(self.policy)}"
                )

        self.episode_rewards = total_rewards[-n_episodes:]
        self.episode_lengths = []
        self.episode_count = iteration + 1

        print("=" * 70)
        print(f"Обучение CEM завершено! Итераций: {self.episode_count}")
        print(f"Политика покрывает {len(self.policy)} состояний")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "q_value_history": [],
        }

    def save(self, filepath: str):
        data = {
            "name": self.name,
            "policy": dict(self.policy),
            "n_actions": self.n_actions,
            "elite_percentile": self.elite_percentile,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"CEM агент сохранён: {filepath}")

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.name = data["name"]
        self.n_actions = data["n_actions"]
        self.elite_percentile = data.get("elite_percentile", 0.1)
        self.policy = defaultdict(
            lambda: np.ones(self.n_actions) / self.n_actions,
            {k: np.array(v) for k, v in data["policy"].items()},
        )
        print(f"CEM агент загружен: {filepath}")
