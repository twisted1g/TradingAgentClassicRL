from envs.trading_env import MyTradingEnv
import pandas as pd
import numpy as np
from agents.classical.qlearning_agent import QLearningAgent
from agents.classical.monte_carlo_agent import MonteCarloAgent
from agents.classical.sarsa_agent import SarsaAgent
from agents.classical.sarsa_lambda_agent import SarsaLambdaAgent

print("=" * 70)
print("ТЕСТИРОВАНИЕ КЛАССИЧЕСКИХ RL АГЕНТОВ")
print("=" * 70)

data_path = "data/data_1h.csv"
df = pd.read_csv(
    data_path,
    index_col=0,
    parse_dates=True,
    date_format="iso8601",
)
df1 = df.iloc[:4000]
df2 = df.iloc[4000:]

env = MyTradingEnv(
    df=df1,
    initial_balance=1000.0,
    window_size=10,
    max_holding_time=336,
)

print(f"\nСреда создана:")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")
print(f"  Данные для обучения: {len(df1)} баров")
print(f"  Данные для тестирования: {len(df2)} баров")

agents = [
    # QLearningAgent(),
    # MonteCarloAgent(),
    # SarsaAgent(),
    SarsaLambdaAgent(lambda_param=0.5),
]

results = {}

print(f"\n{'='*70}")
print("ОБУЧЕНИЕ АГЕНТОВ")
print("=" * 70)

for agent in agents:
    print(f"\nОбучение: {agent.name}")
    print("-" * 50)

    history = agent.train(env, n_episodes=4000, verbose=True)
    results[agent.name] = {"history": history, "agent": agent}

print(f"\n{'='*70}")
print("ТЕСТИРОВАНИЕ НА НОВЫХ ДАННЫХ")
print("=" * 70)

test_env = MyTradingEnv(
    df=df2,
    initial_balance=1000.0,
    window_size=10,
    max_holding_time=336,
)

print(f"\n{'='*70}")
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
print("=" * 70)
print(
    f"{'Агент':<25} {'Доходность':<12} {'Сделки':<8} {'Win Rate':<10} {'Ср.PnL':<10} {'Q-таблица':<12}"
)
print("-" * 90)

best_agent = None
best_profit = -float("inf")

for agent_name, result in results.items():
    agent = result["agent"]

    state, _ = test_env.reset()
    done = False
    total_reward = 0
    actions_taken = []

    while not done:
        action = agent.select_action(state, training=False)
        actions_taken.append(action)
        state, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        done = terminated or truncated

    metrics = test_env.get_metrics()

    final_portfolio = test_env._portfolio_value
    profit_percent = ((final_portfolio / test_env.initial_balance) - 1) * 100

    print(
        f"{agent_name:<25} {profit_percent:>7.2f}% {metrics['total_trades']:>8} {metrics['win_rate']:>9.1f}% ${metrics['avg_pnl']:>8.2f} {len(agent.q_table):>10}"
    )

    results[agent_name]["test_metrics"] = {
        "profit_percent": profit_percent,
        "final_portfolio": final_portfolio,
        "total_trades": metrics["total_trades"],
        "win_rate": metrics["win_rate"],
        "avg_pnl": metrics["avg_pnl"],
        "total_pnl": metrics["total_pnl"],
        "actions_taken": actions_taken,
    }

    if profit_percent > best_profit:
        best_profit = profit_percent
        best_agent = agent_name

print("-" * 90)
print(f"🏆 ЛУЧШИЙ АГЕНТ: {best_agent} ({best_profit:.2f}%)")

print(f"\n{'='*70}")
print("ДЕТАЛЬНАЯ СТАТИСТИКА ДЕЙСТВИЙ")
print("=" * 70)

action_names = {0: "Hold Flat", 1: "Open Long", 2: "Close Long"}

for agent_name, result in results.items():
    actions_taken = result["test_metrics"]["actions_taken"]
    action_counts = pd.Series(actions_taken).value_counts().sort_index()

    print(f"\n{agent_name}:")
    total_actions = len(actions_taken)
    for act in [0, 1, 2]:
        count = action_counts.get(act, 0)
        pct = count / total_actions * 100
        print(f"  {action_names[act]}: {count:>4} раз ({pct:5.1f}%)")

print(f"\n{'='*70}")
print("СРАВНЕНИЕ МЕТРИК ОБУЧЕНИЯ")
print("=" * 70)
print(f"{'Агент':<25} {'Ср.награда':<12} {'Эпизоды':<10} {'Размер Q':<12}")
print("-" * 70)

for agent_name, result in results.items():
    history = result["history"]
    episode_rewards = history["episode_rewards"]

    if len(episode_rewards) >= 50:
        avg_reward = np.mean(episode_rewards[-50:])
    else:
        avg_reward = np.mean(episode_rewards)

    agent = result["agent"]

    print(
        f"{agent_name:<25} {avg_reward:>10.2f} {agent.episode_count:>10} {len(agent.q_table):>10}"
    )

print(f"\n{'='*70}")
print("АНАЛИЗ ТОРГОВОЙ СТРАТЕГИИ ЛУЧШЕГО АГЕНТА")
print("=" * 70)

if best_agent:
    best_result = results[best_agent]
    metrics = best_result["test_metrics"]

    print(f"Агент: {best_agent}")
    print(f"Финальный портфель: ${metrics['final_portfolio']:.2f}")
    print(f"Начальный баланс: ${test_env.initial_balance:.2f}")
    print(f"Доходность: {metrics['profit_percent']:.2f}%")
    print(f"Общий PnL: ${metrics['total_pnl']:.2f}")
    print(f"Количество сделок: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Средний PnL на сделку: ${metrics['avg_pnl']:.2f}")

    actions_taken = metrics["actions_taken"]
    total_steps = len(actions_taken)
    trading_activity = (actions_taken.count(1) / total_steps) * 100
    print(f"Активность торговли: {trading_activity:.1f}%")

print(f"\n{'='*70}")
print("РЕКОМЕНДАЦИИ")
print("=" * 70)


print("Анализ переобучения:")
for agent_name, result in results.items():
    train_history = result["history"]["episode_rewards"]
    test_profit = result["test_metrics"]["profit_percent"]

    if len(train_history) >= 50:
        final_train_perf = np.mean(train_history[-50:])
        if final_train_perf > 0 and test_profit < -5:
            print(f"  {agent_name}: возможное переобучение")
        elif test_profit > 0:
            print(f"  {agent_name}: стабильная работа")
        else:
            print(f"  {agent_name}: низкая эффективность")
