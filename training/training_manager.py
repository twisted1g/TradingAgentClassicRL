import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
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

    def _create_validation_split(self, df: pd.DataFrame, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Разделяет данные на train и validation"""
        split_idx = int(len(df) * (1 - val_ratio))
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        return train_df, val_df

    def _create_envs(self, df: pd.DataFrame, config: TrainingConfig) -> Tuple[MyTradingEnv, MyTradingEnv]:
        """Создает train и validation окружения"""
        train_df, val_df = self._create_validation_split(df, val_ratio=0.2)
        
        train_env = MyTradingEnv(
            df=train_df.copy(),
            initial_balance=config.initial_balance,
            window_size=config.window_size,
            commission=config.commission,
            slippage=config.slippage,
            max_holding_time=config.max_holding_time,
            holding_threshold=config.holding_threshold,
            max_drawdown_threshold=config.max_drawdown_threshold,
            lambda_drawdown=config.lambda_drawdown,
            lambda_hold=config.lambda_hold,
            reward_scaling=config.reward_scaling,
            max_steps=config.max_steps,
        )
        
        val_env = MyTradingEnv(
            df=val_df.copy(),
            initial_balance=config.initial_balance,
            window_size=config.window_size,
            commission=config.commission,
            slippage=config.slippage,
            max_holding_time=config.max_holding_time,
            holding_threshold=config.holding_threshold,
            max_drawdown_threshold=config.max_drawdown_threshold,
            lambda_drawdown=config.lambda_drawdown,
            lambda_hold=config.lambda_hold,
            reward_scaling=config.reward_scaling,
            max_steps=config.max_steps,
        )
        
        return train_env, val_env

    def _evaluate_agent(self, agent, env: MyTradingEnv, n_episodes: int = 5) -> Dict[str, Any]:
        """Оценка агента на отдельном окружении"""
        rewards = []
        portfolio_values = []
        trade_counts = []
        
        for i in range(n_episodes):
            state, _ = env.reset(seed=self.seed + i + 1000)  # Фиксированные seeds для оценки
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.select_action(state, training=False)
                state, reward, done, _, info = env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
            portfolio_values.append(info.get("portfolio_value", env.initial_balance))
            trade_counts.append(info.get("n_trades", 0))
        
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_portfolio": float(np.mean(portfolio_values)),
            "mean_trades": float(np.mean(trade_counts)),
        }

    def _print_progress(self, episode: int, total_episodes: int, episode_reward: float,
                       agent, val_results: Dict[str, Any], env_metrics: Dict[str, Any]):
        """Печать прогресса обучения"""
        # Средняя награда за последние 100 эпизодов
        if len(agent.episode_rewards) >= 100:
            avg_reward = np.mean(agent.episode_rewards[-100:])
        else:
            avg_reward = episode_reward
        
        print(f"Ep {episode+1:5d}/{total_episodes} | "
              f"Train R: {episode_reward:7.2f} | "
              f"Avg100 R: {avg_reward:7.2f} | "
              f"Val R: {val_results['mean_reward']:7.2f} ± {val_results['std_reward']:5.2f} | "
              f"Val Portf: ${val_results['mean_portfolio']:8.2f} | "
              f"Eps: {agent.epsilon:6.4f} | "
              f"LR: {getattr(agent, 'learning_rate', 0.1):6.4f} | "
              f"States: {len(agent.q_table):6d} | "
              f"Trades: {env_metrics.get('total_trades', 0):3d}")

    def train_agent(
        self,
        agent,
        df: pd.DataFrame,
        config: TrainingConfig,
        experiment_name: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Основной метод обучения агента"""
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{config.agent_name}_{timestamp}"
        
        logger = TrainingLogger(self.base_log_dir, experiment_name)
        checkpoint_dir = self.base_checkpoint_dir / experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем конфиг
        with open(logger.log_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Начало обучения: {config.agent_name}")
            print(f"Эксперимент: {experiment_name}")
            print(f"Эпизодов: {config.n_episodes}")
            print(f"Max steps: {config.max_steps}")
            print(f"Learning rate: {config.learning_rate}")
            print(f"Discount factor: {config.discount_factor}")
            print(f"Epsilon: {config.epsilon_start}->{config.epsilon_end}")
            print(f"{'='*80}\n")
        
        # Создаем окружения
        train_env, val_env = self._create_envs(df, config)
        
        # Метрики для мониторинга
        best_val_reward = -float('inf')
        patience_counter = 0
        start_time = time.time()
        
        # Основной цикл обучения
        for episode in range(config.n_episodes_start, config.n_episodes):
            # 1. ОБУЧЕНИЕ НА TRAIN ДАННЫХ
            state, info = train_env.reset(seed=self.seed + episode)
            done = False
            episode_reward = 0
            steps = 0
            
            # Для SARSA и Monte Carlo нужна специальная обработка
            if hasattr(agent, 'reset_eligibility_traces'):
                agent.reset_eligibility_traces()
            
            if hasattr(agent, '_start_new_episode'):
                agent._start_new_episode()
            
            action = agent.select_action(state, training=True)
            
            while not done and steps < config.max_steps:
                next_state, reward, terminated, truncated, info = train_env.step(action)
                done = terminated or truncated
                
                # Правильная обработка next_action для разных алгоритмов
                if not done:
                    next_action = agent.select_action(next_state, training=True)
                else:
                    next_action = None
                
                # Обновление агента
                agent.update(state, action, reward, next_state, done, next_action)
                
                # Переход к следующему шагу
                state = next_state
                if next_action is not None:
                    action = next_action
                else:
                    action = agent.select_action(state, training=True)
                
                episode_reward += reward
                steps += 1
            
            # Завершение эпизода для Monte Carlo
            if hasattr(agent, '_end_episode'):
                agent._end_episode()
            
            # 2. ОБНОВЛЕНИЕ МЕТРИК АГЕНТА
            agent.episode_count += 1
            agent.episode_rewards.append(episode_reward)
            agent.episode_lens.append(steps)
            
            # Decay epsilon
            if hasattr(agent, 'decay_epsilon'):
                agent.decay_epsilon()
            
            # Decay learning rate
            if hasattr(agent, 'decay_learning_rate'):
                agent.decay_learning_rate()
            
            # 3. СОБИРАЕМ МЕТРИКИ ОКРУЖЕНИЯ
            env_metrics = train_env.get_metrics()
            portfolio_value = info.get("portfolio_value", train_env.portfolio_value)
            
            # Создаем объект метрик эпизода
            episode_metrics = EpisodeMetrics(
                episode=episode + 1,
                reward=float(episode_reward),
                steps=int(steps),
                epsilon=float(getattr(agent, 'epsilon', 0.0)),
                portfolio_value=float(portfolio_value),
                n_trades=int(env_metrics.get("total_trades", 0)),
                win_rate=float(env_metrics.get("win_rate", 0)),
                avg_pnl=float(env_metrics.get("avg_pnl", 0)),
                max_drawdown=float(env_metrics.get("max_drawdown", 0)),
                timestamp=time.time(),
            )
            
            # Логируем эпизод
            logger.log_episode(episode_metrics)
            
            # 4. ВАЛИДАЦИЯ КАЖДЫЕ eval_frequency ЭПИЗОДОВ
            if (episode + 1) % config.eval_frequency == 0:
                val_results = self._evaluate_agent(agent, val_env, n_episodes=3)
                logger.log_evaluation(episode + 1, val_results)
                
                # Проверяем улучшение
                if val_results['mean_reward'] > best_val_reward:
                    best_val_reward = val_results['mean_reward']
                    patience_counter = 0
                    # Сохраняем лучшую модель
                    agent.save(str(checkpoint_dir / "best_agent.pkl"))
                    if verbose:
                        print(f"✓ Новая лучшая модель! Val reward: {best_val_reward:.2f}")
                else:
                    patience_counter += 1
                
                # Печать прогресса
                if verbose:
                    self._print_progress(episode, config.n_episodes, episode_reward, 
                                        agent, val_results, env_metrics)
            
            # 5. СОХРАНЕНИЕ ЧЕКПОЙНТОВ
            if (episode + 1) % config.save_frequency == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_ep{episode+1}.pkl"
                agent.save(str(checkpoint_path))
                logger.log_checkpoint(episode + 1, str(checkpoint_path))
            
            # 6. РАННЯЯ ОСТАНОВКА
            if patience_counter >= config.patience:
                if verbose:
                    print(f"\n⚠️  Ранняя остановка на эпизоде {episode + 1}")
                    print(f"   Эпизодов без улучшений: {patience_counter}")
                break
        
        # 7. ФИНАЛИЗАЦИЯ ОБУЧЕНИЯ
        training_time = time.time() - start_time
        
        # Сохраняем финальную модель
        final_path = checkpoint_dir / "final_agent.pkl"
        agent.save(str(final_path))
        logger.log_checkpoint(config.n_episodes, str(final_path))
        
        # Сохраняем итоговый отчет
        logger.save_summary(config, training_time, best_val_reward)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print(f"{'='*80}")
            print(f"Всего эпизодов: {agent.episode_count}")
            print(f"Время обучения: {training_time/60:.2f} минут")
            print(f"Лучшая валидационная награда: {best_val_reward:.2f}")
            print(f"Размер Q-таблицы: {len(agent.q_table)} состояний")
            print(f"Финальный epsilon: {getattr(agent, 'epsilon', 0.0):.4f}")
            
            # Оценка на validation данных
            final_eval = self._evaluate_agent(agent, val_env, n_episodes=10)
            print(f"\nФинальная оценка на validation:")
            print(f"  Средняя награда: {final_eval['mean_reward']:.2f}")
            print(f"  Средний портфель: ${final_eval['mean_portfolio']:.2f}")
            print(f"  Среднее количество сделок: {final_eval['mean_trades']:.1f}")
            print(f"{'='*80}\n")
        
        return {
            "experiment_name": experiment_name,
            "log_dir": str(logger.log_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "final_agent_path": str(final_path),
            "best_agent_path": str(checkpoint_dir / "best_agent.pkl"),
            "training_time": training_time,
            "best_val_reward": best_val_reward,
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
        """Продолжение обучения с загруженного чекпойнта"""
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
            verbose=verbose
        )