import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from .dataclasses import TrainingConfig, EpisodeMetrics
from .training_logger import TrainingLogger


class TrainingManager:
    def __init__(
        self,
        base_log_dir: str = "   training_data/logs",
        base_checkpoint_dir: str = "training_data/checkpoints",
    ):
        self.base_log_dir = Path(base_log_dir)
        self.base_checkpoint_dir = Path(base_checkpoint_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        self.base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_agent(
        self,
        agent,
        env,
        config: TrainingConfig,
        experiment_name: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{config.agent_name}_{timestamp}"

        logger = TrainingLogger(self.base_log_dir, experiment_name)
        checkpoint_dir = self.base_checkpoint_dir / experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with open(logger.log_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        if verbose:
            print(f"\n{'='*70}")
            print(f"Начало обучения: {config.agent_name}")
            print(f"Эксперимент: {experiment_name}")
            print(f"{'='*70}")
            print(f"Эпизодов: {config.n_episodes}")
            print(f"Learning rate: {config.learning_rate}")
            print(f"Discount factor: {config.discount_factor}")
            print(f"{'='*70}\n")

        start_time = time.time()

        for episode in range(config.n_episodes):

            state, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0

            action = agent.select_action(state, training=True)

            while not done and steps < config.max_steps:

                next_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                if not done:
                    next_action = agent.select_action(next_state, training=True)
                else:
                    next_action = 0

                agent.update(state, action, reward, next_state, done, next_action)

                state = next_state
                action = next_action

                episode_reward += reward
                steps += 1

            agent.episode_count += 1
            agent.episode_rewards.append(episode_reward)
            agent.episode_lens.append(steps)
            agent.decay_epsilon()

            env_metrics = env.get_metrics()

            episode_metrics = EpisodeMetrics(
                episode=episode + 1,
                reward=episode_reward,
                steps=steps,
                epsilon=agent.epsilon,
                portfolio_value=env._portfolio_value,
                n_trades=env_metrics.get("total_trades", 0),
                win_rate=env_metrics.get("win_rate", 0),
                avg_pnl=env_metrics.get("avg_pnl", 0),
                max_drawdown=env_metrics.get("max_drawdown", 0),
                timestamp=time.time(),
            )
            logger.log_episode(episode_metrics)

            if (episode + 1) % config.eval_frequency == 0:
                eval_results = agent.evaluate(
                    env, n_episodes=10, max_steps=config.max_steps
                )
                logger.log_evaluation(episode + 1, eval_results)

                if verbose:
                    avg_reward = (
                        np.mean(agent.episode_rewards[-100:])
                        if len(agent.episode_rewards) >= 100
                        else 0
                    )
                    print(f"Эпизод {episode + 1}/{config.n_episodes}")
                    print(
                        f"  Награда: {episode_reward:.2f} | Средняя (100): {avg_reward:.2f}"
                    )
                    print(
                        f"  Оценка: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}"
                    )
                    print(
                        f"  Epsilon: {agent.epsilon:.4f} | Состояний: {len(agent.q_table)}"
                    )
                    print(
                        f"  Сделок: {env_metrics.get('total_trades', 0)} | Win Rate: {env_metrics.get('win_rate', 0):.1f}%"
                    )
                    print(f"  Портфель: ${env._portfolio_value:.2f}")
                    print()

            if (episode + 1) % config.save_frequency == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_ep{episode+1}.pkl"
                agent.save(str(checkpoint_path))
                logger.log_checkpoint(episode + 1, str(checkpoint_path))

        training_time = time.time() - start_time

        final_path = checkpoint_dir / "final_agent.pkl"
        agent.save(str(final_path))
        logger.log_checkpoint(config.n_episodes, str(final_path))

        logger.save_summary(config, training_time)

        if verbose:
            print(f"\n{'='*70}")
            print(f"Обучение завершено!")
            print(f"Время обучения: {training_time/60:.2f} минут")
            print(f"Финальная награда: {episode_metrics.reward:.2f}")
            print(f"Размер Q-таблицы: {len(agent.q_table)} состояний")
            print(f"{'='*70}\n")

        return {
            "experiment_name": experiment_name,
            "log_dir": str(logger.log_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "final_agent_path": str(final_path),
            "training_time": training_time,
            "final_metrics": episode_metrics.to_dict(),
        }
