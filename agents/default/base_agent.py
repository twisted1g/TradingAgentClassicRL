from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseAgent(ABC):
    def __init__(
        self,
        name: str = "BaseAgent",
        initial_balance: int = 1000,
        seed: Optional[int] = None,
    ):
        self.name = name
        self.initial_balance = initial_balance
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.reset()

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,
        info: Dict[str, Any],
    ) -> int:
        pass

    def reset(self):
        self.position = 0
        self.entry_price = 0.0
        self.current_step = 0
        self.history = {
            "actions": [],
            "positions": [],
            "prices": [],
            "rewards": [],
        }

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: Dict[str, Any],
    ):
        self.history["actions"].append(action)
        self.history["positions"].append(self.position)
        self.history["rewards"].append(reward)

        if "current_price" in info:
            self.history["prices"].append(info["current_price"])

        self.current_step += 1

    def get_status(self) -> Dict[str, Any]:
        actions = np.array(self.history["actions"])
        rewards = np.array(self.history["rewards"])

        stats = {
            "agent_name": self.name,
            "total_steps": len(actions),
            "total_reward": np.sum(rewards),
            "mean_reward": np.mean(rewards) if len(rewards) > 0 else 0,
            "num_buys": np.sum(actions == 1),
            "num_sells": np.sum(actions == 2),
            "num_holds": np.sum(actions == 0),
        }

        return stats

    def evaluate(
        self,
        env,
        n_episodes: int = 1,
        render: bool = False,
        verbose: bool = True,
        render_interval: int = 50,
    ) -> Dict[str, Any]:
        all_rewards = []
        all_final_values = []

        for episode in range(n_episodes):
            self.reset()
            obs, info = env.reset()
            done = False
            episode_reward = 0
            step = 0

            while not done:
                action = self.act(observation=obs, info=info)
                next_obs, reward, terminated, truncated, info = env.step(action=action)
                done = terminated or truncated

                self.update(
                    observation=obs,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    done=done,
                    info=info,
                )
                episode_reward += reward
                obs = next_obs
                step += 1

                if render and step % render_interval == 0:
                    env.render()

            all_rewards.append(episode_reward)
            all_final_values.append(info.get("portfolio_value", 0))

            if verbose:
                print(
                    f"Episode {episode + 1} / {n_episodes}: "
                    f"Reward={episode_reward:.2f}, "
                    f"Final Value=${info.get('portfolio_value', 0):.2f}"
                )

        stats = self.get_status()
        stats.update(
            {
                "n_episodes": n_episodes,
                "mean_episode_reward": float(np.mean(all_rewards)),
                "std_episode_reward": (
                    float(np.std(all_rewards)) if len(all_rewards) > 1 else 0.0
                ),
                "mean_final_value": float(np.mean(all_final_values)),
                "std_final_value": (
                    float(np.std(all_final_values))
                    if len(all_final_values) > 1
                    else 0.0
                ),
            }
        )

        return stats

    def __str__(self) -> str:
        return f"{self.name} Position: {self.position}, Step: {self.current_step}"

    def __repr__(self) -> str:
        return self.__str__()
