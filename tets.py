from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
from envs.trading_env import MyTradingEnv
import warnings
warnings.filterwarnings('ignore')

class ImprovedTradingEnv(MyTradingEnv):
    """
    Улучшенная версия среды с лучшими наградами и наблюдениями
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(10,),
            dtype=np.float32
        )
        
    def _get_observation(self) -> np.ndarray:
        if self._idx >= len(self.df):
            return np.zeros(10)
        
        row = self.df.iloc[self._idx]
        
        # Используем безопасное получение значений
        rsi = row.get("rsi", 50) / 100.0
        macd_diff = row.get("macd_diff", 0)
        price_to_ma20 = row.get("price_to_ma20", 0)
        price_trend = row.get("price_trend", 0)
        returns = row.get("returns", 0)
        
        ma_20 = row.get("ma_20", row["close"])
        ma_50 = row.get("ma_50", row["close"])
        
        observation = np.array([
            rsi,
            macd_diff * 100,
            price_to_ma20 * 100,
            price_trend / max(row["close"], 0.001) * 100,
            returns * 100,
            (row["close"] - ma_20) / max(ma_20, 0.001) * 100,
            (row["close"] - ma_50) / max(ma_50, 0.001) * 100,
            self._position,
            self.current_holding_time / max(self.max_holding_time, 1),
            min(self.max_drawdown, 1.0)  # Ограничиваем просадку
        ], dtype=np.float32)
        
        # Заменяем NaN и бесконечности
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        
        return observation
    
    def _calculate_reward(self, prev_portfolio_value: float) -> float:
        # Базовая награда за изменение портфеля (в процентах)
        if prev_portfolio_value > 0:
            portfolio_return = (self._portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            portfolio_return = 0
            
        base_reward = portfolio_return * 1000  # Увеличиваем масштаб
        
        # Бонусы/штрафы
        trade_bonus = 0.0
        hold_penalty = 0.0
        
        # Бонус за успешную сделку
        if self._position == 1 and self.current_holding_time == 1:
            trade_bonus = 0.5
            
        # Штраф за просадку (только если в позиции)
        if self._position == 1:
            drawdown_penalty = -self.lambda_drawdown * self.max_drawdown * 50
        else:
            drawdown_penalty = 0
            
        # Штраф за слишком частые сделки
        if len(self.trade_history) > 0 and self.current_holding_time < 3:
            frequency_penalty = -0.1
        else:
            frequency_penalty = 0
            
        total_reward = base_reward + trade_bonus + hold_penalty + drawdown_penalty + frequency_penalty
        
        return total_reward * self.reward_scaling

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, Dict[str, Any]]:
        try:
            return super().step(action)
        except Exception as e:
            print(f"Ошибка в step: {e}")
            # Возвращаем безопасное состояние при ошибке
            obs = self._get_observation()
            return obs, -10, True, False, {}

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.logger.record('train/avg_episode_reward', avg_reward)
        
        if len(self.training_env.envs[0].unwrapped.trade_history) > 0:
            metrics = self.training_env.envs[0].unwrapped.get_metrics()
            self.logger.record('trading/total_trades', metrics['total_trades'])
            self.logger.record('trading/win_rate', metrics['win_rate'])
            self.logger.record('trading/avg_pnl', metrics['avg_pnl'])
            self.logger.record('trading/portfolio_value', 
                             self.training_env.envs[0].unwrapped._portfolio_value)
            
        return True

def plot_results(portfolio_values, actions_history, positions_history, current_prices, metrics):
    """
    Визуализация результатов тестирования
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # График стоимости портфеля
    ax1.plot(portfolio_values, label='Портфель', color='blue', linewidth=2)
    ax1.axhline(y=10000, color='red', linestyle='--', alpha=0.7, label='Начальный баланс')
    ax1.set_title('Динамика стоимости портфеля')
    ax1.set_xlabel('Шаг')
    ax1.set_ylabel('Стоимость ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # График цен
    ax2.plot(current_prices, label='Цена', color='green', linewidth=1)
    ax2.set_title('Динамика цены актива')
    ax2.set_xlabel('Шаг')
    ax2.set_ylabel('Цена ($)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # График действий
    # Конвертируем actions_history в простой список чисел
    actions_clean = [a.item() if hasattr(a, 'item') else a for a in actions_history]
    ax3.plot(actions_clean, label='Действия', color='red', alpha=0.7, linewidth=1)
    ax3.set_title('Действия агента')
    ax3.set_xlabel('Шаг')
    ax3.set_ylabel('Действие')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Hold', 'Buy', 'Sell'])
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # График позиций
    positions_clean = [p.item() if hasattr(p, 'item') else p for p in positions_history]
    ax4.plot(positions_clean, label='Позиция', color='purple', alpha=0.7, linewidth=1)
    ax4.set_title('Позиции агента')
    ax4.set_xlabel('Шаг')
    ax4.set_ylabel('Позиция')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Flat', 'Long'])
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Выводим метрики на графике
    if metrics['total_trades'] > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        metric_names = ['Win Rate', 'Avg PnL', 'Avg Holding Time']
        metric_values = [metrics['win_rate'], metrics['avg_pnl'], metrics['avg_holding_time']]
        
        bars = ax.bar(metric_names, metric_values, color=['green', 'blue', 'orange'])
        ax.set_title('Торговые метрики')
        ax.set_ylabel('Значение')
        
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def train_agent(df, total_timesteps=50000):
    print("Создание улучшенной среды...")
    
    env = ImprovedTradingEnv(
        df=df,
        initial_balance=10000.0,
        window_size=10,
        commission=0.0005,  # Увеличиваем комиссию для уменьшения количества сделок
        slippage=0.001,     # Увеличиваем проскальзывание
        max_holding_time=100,
        holding_threshold=20,
        max_drawdown_threshold=0.03,  # Уменьшаем порог просадки
        lambda_drawdown=0.2,  # Увеличиваем штраф за просадку
        lambda_hold=0.05,     # Увеличиваем штраф за долгое удержание
        reward_scaling=5.0,   # Уменьшаем масштаб наград
    )
    
    check_env(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    print("Создание модели PPO с консервативными гиперпараметрами...")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-5,  # Еще меньше learning rate
        n_steps=2048,        # Увеличиваем для лучшей стабильности
        batch_size=128,
        n_epochs=5,          # Уменьшаем количество эпох
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.05,     # Уменьшаем для более консервативного обучения
        ent_coef=0.05,       # Уменьшаем exploration
        vf_coef=0.5,
        max_grad_norm=0.3,   # Уменьшаем для большей стабильности
        tensorboard_log="./tensorboard_logs/",
        verbose=1,
        device='auto'
    )
    
    print("Начало обучения...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=TensorboardCallback(),
        tb_log_name="conservative_trading_ppo",
        progress_bar=True
    )
    
    print("Обучение завершено!")
    return model, env

def test_agent(model, df, initial_balance=10000.0):
    print("\n=== ТЕСТИРОВАНИЕ АГЕНТА ===")
    
    test_env = ImprovedTradingEnv(
        df=df,
        initial_balance=initial_balance,
        window_size=10,
        commission=0.0005,
        slippage=0.001,
        max_holding_time=100,
        holding_threshold=144,
        max_drawdown_threshold=0.05,
        lambda_drawdown=0.5,
        lambda_hold=0.05,
        reward_scaling=5.0,
    )
    
    obs, _ = test_env.reset()
    done = False
    
    portfolio_values = [initial_balance]
    actions_history = []
    positions_history = []
    current_prices = []
    rewards_history = []
    
    step = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # Детерминированно для теста
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        portfolio_values.append(info['portfolio_value'])
        actions_history.append(action)
        positions_history.append(info['position'])
        current_prices.append(info['current_price'])
        rewards_history.append(reward)
        
        step += 1
        if step % 500 == 0:
            current_trades = len(test_env.trade_history)
            print(f"Шаг {step}, Портфель: ${info['portfolio_value']:.2f}, "
                  f"Позиция: {info['position']}, Сделок: {current_trades}")
    
    metrics = test_env.get_metrics()
    
    print("\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===")
    print(f"Начальный баланс: ${initial_balance:.2f}")
    print(f"Конечный баланс: ${portfolio_values[-1]:.2f}")
    print(f"Общая прибыль: ${portfolio_values[-1] - initial_balance:.2f}")
    print(f"Доходность: {(portfolio_values[-1] / initial_balance - 1) * 100:.2f}%")
    print(f"Всего сделок: {metrics['total_trades']}")
    
    if metrics['total_trades'] > 0:
        print(f"Винрейт: {metrics['win_rate']:.2f}%")
        print(f"Средний PnL: ${metrics['avg_pnl']:.4f}")
        print(f"Среднее время удержания: {metrics['avg_holding_time']:.1f} шагов")
        if 'max_drawdown' in metrics:
            print(f"Максимальная просадка: {metrics['max_drawdown'] * 100:.2f}%")
        if 'trades_closed_by_drawdown' in metrics:
            print(f"Сделки закрытые по просадке: {metrics['trades_closed_by_drawdown']}")
        if 'trades_closed_by_time' in metrics:
            print(f"Сделки закрытые по времени: {metrics['trades_closed_by_time']}")
    else:
        print("Нет совершенных сделок")
    
    return portfolio_values, actions_history, positions_history, current_prices, metrics, rewards_history

def analyze_behavior(actions_history, positions_history, rewards_history):
    """Анализ поведения агента"""
    print("\n=== АНАЛИЗ ПОВЕДЕНИЯ АГЕНТА ===")
    
    # Конвертируем в простые Python типы
    actions_clean = [a.item() if hasattr(a, 'item') else int(a) for a in actions_history]
    positions_clean = [p.item() if hasattr(p, 'item') else int(p) for p in positions_history]
    
    action_counts = pd.Series(actions_clean).value_counts().sort_index()
    print("Статистика действий:")
    for action, count in action_counts.items():
        action_name = {0: 'Hold', 1: 'Buy', 2: 'Sell'}[action]
        print(f"  {action_name}: {count} раз ({count/len(actions_clean)*100:.1f}%)")
    
    position_changes = np.diff(positions_clean)
    long_entries = np.sum(position_changes == 1)
    long_exits = np.sum(position_changes == -1)
    
    print(f"Открытий long позиций: {long_entries}")
    print(f"Закрытий long позиций: {long_exits}")
    print(f"Общая награда: {np.sum(rewards_history):.2f}")
    print(f"Средняя награда за шаг: {np.mean(rewards_history):.4f}")
    
    # Анализ паттернов торговли
    if len(actions_clean) > 10:
        action_changes = np.diff(actions_clean)
        unique_patterns = len(set(zip(actions_clean[:-1], actions_clean[1:])))
        print(f"Уникальных паттернов действий: {unique_patterns}")

if __name__ == "__main__":
    print("Загрузка данных...")
    data_path = "data/data_1h_2023.csv"
    train_df = pd.read_csv(data_path, index_col=0, parse_dates=True, date_format="iso8601")
    
    data_path = "data/data_1h_2024.csv"
    test_df = pd.read_csv(data_path, index_col=0, parse_dates=True, date_format="iso8601")
    
    print(f"Размер тренировочных данных: {len(train_df)}")
    print(f"Размер тестовых данных: {len(test_df)}")
    
    # Обучаем агента с консервативными параметрами
    model, env = train_agent(train_df, total_timesteps=10_000)
    
    # Сохраняем модель
    model.save("conservative_trading_ppo_model")
    print("Модель сохранена как 'conservative_trading_ppo_model'")
    
    # Тестируем агента
    portfolio_values, actions_history, positions_history, current_prices, metrics, rewards_history = test_agent(model, test_df)
    
    # Анализируем поведение
    analyze_behavior(actions_history, positions_history, rewards_history)
    
    # Визуализируем результаты
    plot_results(portfolio_values, actions_history, positions_history, current_prices, metrics)