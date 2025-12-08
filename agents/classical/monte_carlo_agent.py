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

        self.q_table: Dict = defaultdict(lambda: np.zeros(self.n_actions))

        self.returns_count: Dict[Tuple, int] = defaultdict(int)

        self.current_episode_transitions: List[Tuple[tuple, int, float]] = []
        self._episode_started = False

    def _start_new_episode(self):

        self.current_episode_transitions = []
        self._episode_started = True

    def _add_transition(self, state: np.ndarray, action: int, reward: float):

        state_key = self.state_to_key(state)
        self.current_episode_transitions.append((state_key, action, float(reward)))

    def _end_episode(self):
        if not self.current_episode_transitions:
            self._episode_started = False
            return

        G = 0.0
        visited_sa: Set[Tuple] = set()

        for t in range(len(self.current_episode_transitions) - 1, -1, -1):
            state_key, action, reward = self.current_episode_transitions[t]
            sa = (state_key, action)

            G = reward + self.discount_factor * G

            if self.return_clip is not None:
                low, high = self.return_clip
                G = float(np.clip(G, low, high))

            if self.first_visit and sa in visited_sa:
                continue

            visited_sa.add(sa)

            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.n_actions)

            current_q = float(self.q_table[state_key][action])

            if self.use_sample_average:

                self.returns_count[sa] += 1
                n = self.returns_count[sa]
                step = 1.0 / n
                new_q = current_q + step * (G - current_q)
            else:

                alpha = float(self.learning_rate)
                new_q = current_q + alpha * (G - current_q)

            self.q_table[state_key][action] = new_q

        self._episode_started = False

        self.current_episode_transitions = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:

        return super().select_action(state, training)

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: Optional[int] = None,
    ):

        if not self._episode_started:
            self._start_new_episode()

        self._add_transition(state, action, reward)

        if done:

            episode_states = set([s for s, _, _ in self.current_episode_transitions])

            self._end_episode()

            if episode_states:
                try:
                    avg_q = np.mean(
                        [
                            np.max(self.q_table[s])
                            for s in episode_states
                            if s in self.q_table
                        ]
                    )
                    self.q_value_history.append(float(avg_q))
                except Exception:

                    pass

    def train_episode(
        self, env, max_steps: int = 1000, verbose: bool = False
    ) -> Dict[str, float]:

        self._start_new_episode()

        state, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        action = self.select_action(state, training=True)

        while not done and steps < max_steps:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self._add_transition(state, action, reward)

            if not done:
                next_action = self.select_action(next_state, training=True)
            else:
                next_action = 0

            state = next_state
            action = next_action

            episode_reward += reward
            steps += 1
            self.total_steps += 1

            if verbose and steps % 100 == 0:
                print(
                    f"Step {steps}: Reward={reward:.4f}, Portfolio={info.get('portfolio_value', 0):.2f}"
                )

        episode_states = set([s for s, _, _ in self.current_episode_transitions])

        self._end_episode()

        if episode_states:
            try:
                avg_q = np.mean(
                    [
                        np.max(self.q_table[s])
                        for s in episode_states
                        if s in self.q_table
                    ]
                )
                self.q_value_history.append(float(avg_q))
            except Exception:
                pass

        self.episode_count += 1
        self.episode_rewards.append(float(episode_reward))
        self.episode_lens.append(int(steps))
        self.decay_epsilon()

        return {
            "episode": self.episode_count,
            "reward": float(episode_reward),
            "steps": int(steps),
            "epsilon": float(self.epsilon),
            "portfolio_value": float(info.get("portfolio_value", 0)),
            "n_trades": int(info.get("n_trades", 0)),
        }

    def save(self, filepath: str):
        import pickle

        data = {
            "name": self.name,
            "q_table": dict(self.q_table),
            "n_actions": self.n_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "first_visit": self.first_visit,
            "use_sample_average": self.use_sample_average,
            "returns_count": dict(self.returns_count),
            "episode_count": self.episode_count,
            "episode_rewards": self.episode_rewards,
            "episode_lens": self.episode_lens,
            "q_value_history": self.q_value_history,
            "total_steps": self.total_steps,
            "agent_type": "MonteCarlo",
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Monte Carlo агент сохранён: {filepath}")

    def load(self, filepath: str):
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        if data.get("agent_type") != "MonteCarlo":
            print("Внимание: Загружаемый файл может быть другого типа агента")

        self.name = data.get("name", self.name)

        q_table_in = data.get("q_table", {})
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions),
            {k: np.array(v, dtype=float) for k, v in q_table_in.items()},
        )
        self.learning_rate = data.get("learning_rate", self.learning_rate)
        self.discount_factor = data.get("discount_factor", self.discount_factor)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.epsilon_start = data.get("epsilon_start", self.epsilon_start)
        self.epsilon_end = data.get("epsilon_end", self.epsilon_end)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        self.first_visit = data.get("first_visit", self.first_visit)
        self.use_sample_average = data.get(
            "use_sample_average", self.use_sample_average
        )

        rc = data.get("returns_count", {})
        self.returns_count = defaultdict(int, {tuple(k): int(v) for k, v in rc.items()})
        self.episode_count = data.get("episode_count", 0)
        self.episode_rewards = data.get("episode_rewards", [])
        self.episode_lens = data.get("episode_lens", [])
        self.q_value_history = data.get("q_value_history", [])
        self.total_steps = data.get("total_steps", 0)

        print(f"Monte Carlo агент загружен: {filepath}")
        print(f"  Эпизодов: {self.episode_count}")
        print(f"  Размер Q-таблицы: {len(self.q_table)}")
        print(
            f"  First-visit: {self.first_visit}, sample-average: {self.use_sample_average}"
        )
