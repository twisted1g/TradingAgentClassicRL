from envs.trading_env import MyTradingEnv
import pandas as pd
from agents.classical.qlearning_agent import QLearningAgent
from agents.classical.cross_entropy_agent import CrossEntropyAgent

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
# df = df.iloc[:1000]


env = MyTradingEnv(
    df=df,
    initial_balance=1000.0,
    window_size=10,
    max_holding_time=50,
)

print(f"\nСреда создана:")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")
print(f"  Данные: {len(df)} баров")

agents = [
    QLearningAgent(n_actions=3, learning_rate=0.1, epsilon_decay=0.999),
    # CrossEntropyAgent(n_actions=3),
    # SARSAAgent(n_actions=3, learning_rate=0.1, epsilon_decay=0.995),
    # ExpectedSARSAAgent(n_actions=3, learning_rate=0.1, epsilon_decay=0.995),
    # MonteCarloAgent(n_actions=3, learning_rate=0.1, epsilon_decay=0.995),
]

results = {}
for agent in agents:
    print(f"\n{'='*70}")
    print(f"Обучение: {agent.name}")
    print("=" * 70)

    history = agent.train(env, n_episodes=2000, verbose=True)

    print(f"\nОценка агента...")
    eval_stats = agent.evaluate(env, n_episodes=10)
    print(
        f"Средняя награда: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}"
    )

    results[agent.name] = {"history": history, "eval": eval_stats}


print("\n" + "=" * 70)
print("СРАВНЕНИЕ АГЕНТОВ")
print("=" * 70)
print(f"{'Агент':<20} {'Средняя награда':<20} {'Размер Q-таблицы':<20}")
print("-" * 70)


test_env = MyTradingEnv(
    df=df,
    initial_balance=1000.0,
    window_size=10,
    max_holding_time=50,
)

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

print(f"Финальный портфель: ${test_env._portfolio_value:.2f}")
print(f"Начальный баланс: ${test_env.initial_balance:.2f}")
print(
    f"Изменение: {((test_env._portfolio_value / test_env.initial_balance) - 1) * 100:.2f}%"
)
print(f"Общая награда (должна совпадать): {total_reward:.2f}")
print(f"Всего шагов: {test_env._idx - test_env.window_size}")
print(f"Количество сделок: {metrics['total_trades']}")
print(f"Win rate: {metrics['win_rate']:.1f}%")
print(f"Средний PnL на сделку: ${metrics['avg_pnl']:.2f}")
print(f"Общий PnL: ${metrics['total_pnl']:.2f}")
print(f"Закрыто по проседанию: {metrics['trades_closed_by_drawdown']}")
print(f"Закрыто по таймауту: {metrics['trades_closed_by_time']}")
print(f"Уникальных действий: {sorted(set(actions_taken))}")


action_counts = pd.Series(actions_taken).value_counts().sort_index()
action_names = {0: "Hold Flat", 1: "Open Long", 2: "Close Long"}
print("\nРаспределение действий:")
for act in [0, 1, 2]:
    count = action_counts.get(act, 0)
    pct = count / len(actions_taken) * 100
    print(f"  {action_names[act]} ({act}): {count} раз ({pct:.1f}%)")


for agent in agents:
    eval_reward = results[agent.name]["eval"]["mean_reward"]
    q_table_size = len(agent.q_table)
    print(f"{agent.name:<20} {eval_reward:<20.2f} {q_table_size:<20}")

print("=" * 70)
