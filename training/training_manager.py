import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import asdict
from .dataclasses import TrainingConfig, EpisodeMetrics
from .training_logger import TrainingLogger
from envs.trading_env import MyTradingEnv


class TrainingManager:
    def __init__(
        self,
        base_log_dir: str = "training_data/logs",
        base_checkpoint_dir: str = "training_data/checkpoints",
        seed: int = 42,
    ):
        self.base_log_dir = Path(base_log_dir)
        self.base_checkpoint_dir = Path(base_checkpoint_dir)
        self.seed = seed
        np.random.seed(seed)

        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        self.base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_env(self, df: pd.DataFrame, config: TrainingConfig) -> MyTradingEnv:
        env_kwargs = {
            "initial_balance": config.initial_balance,
            "window_size": config.window_size,
            "commission": config.commission,
            "slippage": config.slippage,
            "max_holding_time": config.max_holding_time,
            "max_drawdown_threshold": config.max_drawdown_threshold,
            "max_steps": config.max_steps,
        }

        if config.extra_params:
            env_kwargs.update(config.extra_params)

        env = MyTradingEnv(df=df.copy(), **env_kwargs)

        return env

    def _evaluate_agent(
        self, agent, env: MyTradingEnv, n_episodes: int = 5
    ) -> Dict[str, Any]:
        rewards = []
        portfolio_values = []
        trade_counts = []
        steps_list = []
        total_pnls = []
        win_rates = []
        profit_factors = []
        avg_pnls = []

        max_steps = (
            self._current_config.max_steps
            if hasattr(self, "_current_config") and self._current_config
            else 1000
        )

        for i in range(n_episodes):
            state, _ = env.reset(seed=self.seed + i + 1000)
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < max_steps:
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                steps += 1

            env_metrics = env.get_metrics()

            rewards.append(episode_reward)
            portfolio_values.append(info.get("portfolio_value", env.initial_balance))
            trade_counts.append(info.get("n_trades", 0))
            steps_list.append(steps)
            total_pnls.append(env_metrics.get("total_pnl", 0))
            win_rates.append(env_metrics.get("win_rate", 0))
            profit_factors.append(env_metrics.get("profit_factor", 0))
            avg_pnls.append(env_metrics.get("avg_pnl", 0))

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_portfolio": float(np.mean(portfolio_values)),
            "mean_trades": float(np.mean(trade_counts)),
            "mean_steps": float(np.mean(steps_list)),
            "mean_total_pnl": float(np.mean(total_pnls)),
            "mean_win_rate": float(np.mean(win_rates)),
            "mean_profit_factor": float(np.mean(profit_factors)),
            "mean_avg_pnl": float(np.mean(avg_pnls)),
        }

    def _print_progress(
        self,
        episode: int,
        total_episodes: int,
        episode_reward: float,
        agent,
        val_results: Dict[str, Any],
        env_metrics: Dict[str, Any],
    ):
        if len(agent.episode_rewards) >= 100:
            avg_reward = np.mean(agent.episode_rewards[-100:])
        else:
            avg_reward = episode_reward

        win_rate = val_results.get("mean_win_rate", 0)
        profit_factor = val_results.get("mean_profit_factor", 0)
        total_trades = val_results.get("mean_trades", 0)
        avg_pnl = val_results.get("mean_avg_pnl", 0)
        total_pnl = val_results.get("mean_total_pnl", 0)

        reward_str = f"{episode_reward:+7.2f}"
        avg_reward_str = f"{avg_reward:+7.2f}"
        val_reward_str = f"{val_results['mean_reward']:+7.2f}"

        portfolio_change = (
            val_results["mean_portfolio"] - self._current_config.initial_balance
        )
        portfolio_change_pct = (
            portfolio_change / self._current_config.initial_balance
        ) * 100
        portfolio_str = f"${val_results['mean_portfolio']:,.2f}"
        portfolio_change_str = f"{portfolio_change_pct:+.2f}%"

        progress_pct = ((episode + 1) / total_episodes) * 100
        bar_length = 30
        filled = int(bar_length * (episode + 1) / total_episodes)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"\n{'='*100}")
        print(f"Эпизод {episode+1:5d}/{total_episodes} [{progress_pct:5.1f}%] |{bar}|")
        print(f"{'-'*100}")
        print(f"НАГРАДЫ:")
        print(
            f"   Текущая:     {reward_str:>10} | Средняя (100): {avg_reward_str:>10} | Eval: {val_reward_str:>10} ± {val_results['std_reward']:5.2f}"
        )
        print(f"ПОРТФЕЛЬ:")
        print(
            f"   Значение:    {portfolio_str:>10} | Изменение:     {portfolio_change_str:>10}"
        )
        print(f"ТОРГОВЛЯ (на основе eval):")
        print(
            f"   Сделок:      {total_trades:>10.1f} | Win Rate:      {win_rate:>6.1f}% | Profit Factor: {profit_factor:>6.2f}"
        )
        if total_trades > 0:
            print(
                f"   Avg PnL:      ${avg_pnl:>9.2f} | Total PnL:     ${total_pnl:>9.2f}"
            )
        print(f"ПАРАМЕТРЫ:")
        print(
            f"   Epsilon:      {agent.epsilon:>10.4f} | Learning Rate: {getattr(agent, 'learning_rate', 0.1):>10.4f} | States: {len(agent.q_table):>6d}"
        )
        print(f"{'='*100}")

    def train_agent(
        self,
        agent,
        df: pd.DataFrame,
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
            json.dump(asdict(config), f, indent=2)

        if verbose:
            print(f"\n{'='*100}")
            print(f" НАЧАЛО ОБУЧЕНИЯ")
            print(f"{'='*100}")
            print(f"Агент:          {config.agent_name}")
            print(f"Эксперимент:    {experiment_name}")
            print(f"Эпизодов:       {config.n_episodes}")
            print(f"Max steps:       {config.max_steps}")
            print(f"Learning rate:  {config.learning_rate}")
            print(f"Discount:       {config.discount_factor}")
            print(f"Epsilon:        {config.epsilon_start} → {config.epsilon_end}")
            print(f"Eval frequency: {config.eval_frequency}")
            print(f"Patience:       {config.patience}")
            print(f"Initial balance: ${config.initial_balance:,.2f}")
            print(f"{'='*100}\n")

        env = self._create_env(df, config)

        self._current_config = config

        start_time = time.time()

        for episode in range(config.n_episodes_start, config.n_episodes):
            state, info = env.reset(seed=self.seed + episode)
            done = False
            episode_reward = 0
            steps = 0

            if hasattr(agent, "reset_eligibility_traces"):
                agent.reset_eligibility_traces()

            if hasattr(agent, "_start_new_episode"):
                agent._start_new_episode()

            if hasattr(agent, "episode_transitions"):
                agent.episode_transitions.clear()

            action = agent.select_action(state, training=True)

            while not done and steps < config.max_steps:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if not done:
                    next_action = agent.select_action(next_state, training=True)
                else:
                    next_action = None

                agent.update(state, action, reward, next_state, done, next_action)

                state = next_state
                if next_action is not None:
                    action = next_action
                else:
                    action = agent.select_action(state, training=True)

                episode_reward += reward
                steps += 1

            if hasattr(agent, "_end_episode"):
                agent._end_episode()

            if hasattr(agent, "episode_transitions") and done:
                if agent.episode_transitions:
                    pass

            agent.episode_count += 1
            agent.episode_rewards.append(episode_reward)
            agent.episode_lens.append(steps)

            if hasattr(agent, "decay_epsilon"):
                agent.decay_epsilon()

            if hasattr(agent, "decay_learning_rate"):
                agent.decay_learning_rate()

            env_metrics = env.get_metrics()
            portfolio_value = info.get("portfolio_value", env.portfolio_value)

            episode_metrics = EpisodeMetrics(
                episode=episode + 1,
                reward=float(episode_reward),
                steps=int(steps),
                epsilon=float(getattr(agent, "epsilon", 0.0)),
                portfolio_value=float(portfolio_value),
                n_trades=int(env_metrics.get("total_trades", 0)),
                win_rate=float(env_metrics.get("win_rate", 0)),
                avg_pnl=float(env_metrics.get("avg_pnl", 0)),
                total_pnl=float(env_metrics.get("total_pnl", 0)),
                max_drawdown=float(env_metrics.get("max_drawdown", 0)),
                timestamp=time.time(),
            )

            logger.log_episode(episode_metrics)

            if (episode + 1) % config.eval_frequency == 0:
                val_results = self._evaluate_agent(agent, env, n_episodes=5)
                logger.log_evaluation(episode + 1, val_results)

                if verbose:
                    self._print_progress(
                        episode,
                        config.n_episodes,
                        episode_reward,
                        agent,
                        val_results,
                        env_metrics,
                    )

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
            final_eval = self._evaluate_agent(agent, env, n_episodes=10)
            final_portfolio_change = (
                final_eval["mean_portfolio"] - config.initial_balance
            )
            final_portfolio_change_pct = (
                final_portfolio_change / config.initial_balance
            ) * 100

            print(f"\n{'='*100}")
            print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print(f"{'='*100}")
            print(f"СТАТИСТИКА ОБУЧЕНИЯ:")
            print(f"   Всего эпизодов:     {agent.episode_count:>10}")
            print(f"   Время обучения:      {training_time/60:>10.2f} минут")
            print(f"   Размер Q-таблицы:   {len(agent.q_table):>10} состояний")
            print(f"   Финальный epsilon:  {getattr(agent, 'epsilon', 0.0):>10.4f}")
            print(f"\nФИНАЛЬНАЯ ОЦЕНКА:")
            print(f"   Средняя награда:     {final_eval['mean_reward']:>+10.2f}")
            print(
                f"   Средний портфель:    ${final_eval['mean_portfolio']:>9,.2f} ({final_portfolio_change_pct:+.2f}%)"
            )
            print(f"   Количество сделок:   {final_eval['mean_trades']:>10.1f}")
            print(f"   Средние шаги:        {final_eval['mean_steps']:>10.1f}")
            print(f"   Win Rate:            {final_eval['mean_win_rate']:>9.1f}%")
            print(f"   Profit Factor:       {final_eval['mean_profit_factor']:>9.2f}")
            print(f"   Avg PnL:            ${final_eval['mean_avg_pnl']:>9.2f}")
            print(f"   Total PnL:          ${final_eval['mean_total_pnl']:>9.2f}")
            print(f"\nСОХРАНЕНО:")
            print(f"   Логи:                {logger.log_dir}")
            print(f"   Чекпойнты:           {checkpoint_dir}")
            print(f"{'='*100}\n")

        return {
            "experiment_name": experiment_name,
            "log_dir": str(logger.log_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "final_agent_path": str(final_path),
            "training_time": training_time,
            "final_metrics": episode_metrics.to_dict(),
            "final_evaluation": final_eval,
        }

    def continue_training(
        self,
        agent,
        df: pd.DataFrame,
        config: TrainingConfig,
        checkpoint_path: str,
        experiment_name: Optional[str] = None,
        verbose: bool = True,
    ):
        if verbose:
            print(f"\nЗагрузка агента из {checkpoint_path}")
        agent.load(checkpoint_path)

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{config.agent_name}_continue_{timestamp}"

        return self.train_agent(
            agent=agent,
            df=df,
            config=config,
            experiment_name=experiment_name,
            verbose=verbose,
        )
