from typing import Any, Dict
import numpy as np
from .base_agent import BaseAgent
from envs.trading_env import MyTradingEnv


def evaluate_agent(
    agent: BaseAgent,
    env: MyTradingEnv,
    n_episodes: int = 1,
    render: bool = False,
    verbose: bool = True,
    render_interval: int = 50,
) -> Dict[str, Any]:
    all_rewards = []
    all_final_values = []

    for episode in range(n_episodes):
        agent.reset()
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            action = agent.act(observation=obs, info=info)
            next_obs, reward, terminated, truncated, info = env.step(action=action)
            done = terminated or truncated

            agent.update(
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
                f"Episode {episode + 1} / {n_episodes}:"
                f"Reward={episode_reward:.2f},"
                f"Final Value=${info.get('portfolio_value', 0):.2f}"
            )

    stats = agent.get_status()
    stats.update(
        {
            "n_episodes": n_episodes,
            "mean_episode_reward": np.mean(all_rewards),
            "std_episode_reward": np.std(all_rewards),
            "mean_final_value": np.mean(all_final_values),
            "std_final_value": np.std(all_final_values),
        }
    )

    return stats
