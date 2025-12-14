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
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        lr_decay: float = 0.9999,
        min_learning_rate: float = 0.001,
        name: str = "BaseClassicalAgent",
        exploration_strategy: str = "epsilon_greedy",
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.name = name
        self.exploration_strategy = exploration_strategy
        self.temperature = 1.0

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.state_visit_count = defaultdict(int)
        self.state_action_visit_count = defaultdict(lambda: np.zeros(self.n_actions))

        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lens = []
        self.q_value_history = []
        self.learning_rate_history = []

    def state_to_key(self, state: np.ndarray) -> tuple:
        return tuple(state.astype(int))

    def get_q_value(self, state: np.ndarray, action: int) -> float:
        return self.q_table[self.state_to_key(state)][action]

    def set_q_value(self, state: np.ndarray, action: int, value: float):
        self.q_table[self.state_to_key(state)][action] = value

    def get_best_action(self, state: np.ndarray) -> int:
        state_key = self.state_to_key(state)
        q_values = self.q_table[state_key]

        if self.exploration_strategy == "ucb":
            counts = self.state_action_visit_count[state_key]
            ucb = q_values + np.sqrt(
                2.0 * np.log(self.total_steps + 1) / (counts + 1e-6)
            )
            return int(np.argmax(ucb))

        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state_key = self.state_to_key(state)
        self.state_visit_count[state_key] += 1

        if not training:
            action = self.get_best_action(state)
            self.state_action_visit_count[state_key][action] += 1
            return action

        if self.exploration_strategy == "epsilon_greedy":
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.n_actions)
            else:
                action = self.get_best_action(state)

        elif self.exploration_strategy == "boltzmann":
            q = self.q_table[state_key]
            exp_q = np.exp(q / (self.temperature + 1e-6))
            probs = exp_q / np.sum(exp_q)
            action = int(np.random.choice(self.n_actions, p=probs))

        else:
            action = self.get_best_action(state)

        self.state_action_visit_count[state_key][action] += 1
        return action

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def decay_learning_rate(self):
        self.learning_rate = max(
            self.min_learning_rate,
            self.learning_rate * self.lr_decay,
        )
        self.learning_rate_history.append(self.learning_rate)

    def decay_temperature(self):
        if self.exploration_strategy == "boltzmann":
            self.temperature = max(0.1, self.temperature * 0.999)

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

    def update_adaptive_learning_rate(self, td_error: float):
        if abs(td_error) > 1.0:
            self.learning_rate = min(0.5, self.learning_rate * 1.01)
        elif abs(td_error) < 0.01:
            self.learning_rate = max(
                self.min_learning_rate, self.learning_rate * 0.99
            )

    def train_episode(
        self,
        env: MyTradingEnv,
        max_steps: int = 1000,
        verbose: bool = False,
        training: bool = True,
    ) -> Dict[str, float]:
        state, info = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False

        action = self.select_action(state, training=training)

        while not done and steps < max_steps:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_action = None
            if not done:
                next_action = self.select_action(next_state, training=training)

            if training:
                self.update(state, action, reward, next_state, done, next_action)
                self.total_steps += 1

            episode_reward += reward
            steps += 1

            if done:
                break

            state = next_state
            action = next_action

        if training:
            self.episode_count += 1
            self.episode_rewards.append(float(episode_reward))
            self.episode_lens.append(int(steps))
            self.decay_epsilon()
            self.decay_learning_rate()
            self.decay_temperature()

        if self.q_table:
            avg_q = np.mean([np.max(v) for v in self.q_table.values()])
            self.q_value_history.append(float(avg_q))

        return {
            "episode": self.episode_count if training else 0,
            "reward": float(episode_reward),
            "steps": int(steps),
            "epsilon": float(self.epsilon if training else 0),
            "portfolio_value": float(info.get("portfolio_value", env.initial_balance)),
            "n_trades": int(info.get("n_trades", 0)),
        }

    def evaluate(
        self, env, n_episodes: int = 10, max_steps: int = 1000
    ) -> Dict[str, Any]:
        rewards, lengths = [], []

        for _ in range(n_episodes):
            state, info = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done and steps < max_steps:
                action = self.select_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            lengths.append(steps)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
        }

    def save(self, path: str):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            raise IOError(f"Error saving agent to {path}: {e}")

    @classmethod
    def load(cls, path: str):
        try:
            with open(path, 'rb') as f:
                agent = pickle.load(f)
            return agent
        except Exception as e:
            raise IOError(f"Error loading agent from {path}: {e}")
        
    def __getstate__(self):
        # Копируем словарь объекта
        state = self.__dict__.copy()
        # Преобразуем defaultdict в обычный dict
        state['q_table'] = dict(state['q_table'])
        state['state_visit_count'] = dict(state['state_visit_count'])
        state['state_action_visit_count'] = dict(state['state_action_visit_count'])
        # Для SARSA(λ)
        if 'eligibility_traces' in state:
            state['eligibility_traces'] = dict(state['eligibility_traces'])
        if 'returns_count' in state:
            state['returns_count'] = dict(state['returns_count'])
        return state

    def __setstate__(self, state):
        q_table = state['q_table']
        state['q_table'] = defaultdict(lambda: np.zeros(state.get('n_actions', 3)))
        state['q_table'].update(q_table)

        state['state_visit_count'] = defaultdict(int)
        state['state_visit_count'].update(state.get('state_visit_count', {}))

        sa_visit = state.get('state_action_visit_count', {})
        state['state_action_visit_count'] = defaultdict(lambda: np.zeros(state.get('n_actions', 3)))
        state['state_action_visit_count'].update(sa_visit)

        if 'eligibility_traces' in state:
            traces = state['eligibility_traces']
            state['eligibility_traces'] = defaultdict(lambda: np.zeros(state.get('n_actions', 3)))
            state['eligibility_traces'].update(traces)

        if 'returns_count' in state:
            rc = state['returns_count']
            state['returns_count'] = defaultdict(int)
            state['returns_count'].update(rc)

        self.__dict__.update(state)