from abc import ABC, abstractmethod
from collections import defaultdict
import pickle
import numpy as np
from typing import Any, Dict, Optional
from envs.trading_env import MyTradingEnv


class BaseClassicalAgent(ABC):
    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1,
        epsilon_end=0.01,
        epsilon_decay: float = 0.995,
        name: str = "BaseClassicalAgent",
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.name = name

        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lens = []
        self.q_value_history = []

    def state_to_key(self, state: np.ndarray) -> tuple:
        return tuple(state.astype(int))

    def get_q_value(self, state: np.ndarray, action: int) -> float:
        state_key = self.state_to_key(state=state)
        return self.q_table[state_key][action]

    def set_q_value(self, state: np.ndarray, action: int, value: float):
        state_key = self.state_to_key(state=state)
        self.q_table[state_key][action] = value

    def get_best_action(self, state: np.ndarray) -> int:
        state_key = self.state_to_key(state=state)
        q_values = self.q_table[state_key]

        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]

        return np.random.choice(best_actions)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return self.get_best_action(state=state)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: Optional[int] = None,
    ):
        pass

    def train_episode(
        self, env: MyTradingEnv, max_steps: int = 1000, verbose: bool = False
    ) -> Dict[str, float]:
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
                f"Q-table size={len(self.q_table)}"
            )

        return {
            "episode": self.episode_count,
            "reward": episode_reward,
            "steps": steps,
            "epsilon": self.epsilon,
        }

    def train(
        self, env, n_episodes: int, max_steps: int = 1000, verbose: bool = True
    ) -> Dict[str, list]:
        print(f"Начало обучения {self.name}")
        print(
            f"Параметры: learning_rate={self.learning_rate}, "
            f"discount_factor={self.discount_factor},"
            f"epsilon={self.epsilon_start}->{self.epsilon_end}"
        )
        print("=" * 70)

        for episode in range(n_episodes):
            self.train_episode(env, max_steps, verbose)

        print("=" * 70)
        print(f"Обучение завершено!")
        print(f"Всего эпизодов: {self.episode_count}")
        print(f"Размер Q-таблицы: {len(self.q_table)} состояний")
        print(
            f"Средняя награда (последние 100): {np.mean(self.episode_rewards[-100:]):.2f}"
        )

        metrics = env.get_metrics()
        print(metrics)

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lens,
            "q_value_history": self.q_value_history,
        }

    def evaluate(
        self, env, n_episodes: int = 10, max_steps: int = 1000
    ) -> Dict[str, Any]:
        eval_rewards = []
        eval_lens = []

        for episode in range(n_episodes):
            state, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < max_steps:
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                state = next_state
                episode_reward += reward
                steps += 1

            eval_rewards.append(episode_reward)
            eval_lens.append(steps)

        return {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "min_reward": np.min(eval_rewards),
            "max_reward": np.max(eval_rewards),
            "mean_length": np.mean(eval_lens),
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
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Агент сохранён: {filepath}")

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
        print(f"Агент загружен: {filepath}")

    def get_policy(self, state: np.ndarray) -> np.ndarray:
        state_key = self.state_to_key(state)
        q_values = self.q_table[state_key]

        exp_q = np.exp(q_values - np.max(q_values))
        return exp_q / np.sum(exp_q)
