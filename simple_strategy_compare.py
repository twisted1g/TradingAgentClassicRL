import pandas as pd
import matplotlib.pyplot as plt

from agents.default.random_agent import RandomAgent
from agents.default.buyhold_agent import BuyAndHoldAgent
from agents.default.moving_average_agent import MovingAverageAgent
from envs.trading_env import MyTradingEnv


def run_and_plot_agent(agent, env, name):
    portfolio_history = []
    step_history = []

    obs, info = env.reset()
    done = False
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

        portfolio_history.append(info["portfolio_value"])
        step_history.append(step)

        obs = next_obs
        step += 1

    return step_history, portfolio_history


def main():
    data_path = "data/data_1h.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True, date_format="iso8601")

    # df = df.iloc[:100_000]
    print(f"Загружено {len(df)} строк данных.")

    initial_balance = 1000.0
    window_size = 10
    max_holding_time = 10**8
    max_drawdown_threshold = 1

    # === RandomAgent ===
    env_random = MyTradingEnv(
        df=df,
        initial_balance=initial_balance,
        window_size=window_size,
        max_holding_time=max_holding_time,
        max_drawdown_threshold=max_drawdown_threshold,
    )
    random_agent = RandomAgent(
        seed=42,
        buy_prob=0.2,
        sell_prob=0.2,
        hold_prob=0.6,
    )
    steps_random, portfolio_random = run_and_plot_agent(
        random_agent, env_random, "RandomAgent"
    )
    random_final = portfolio_random[-1]

    eval_random = random_agent.evaluate(
        env=MyTradingEnv(
            df=df,
            initial_balance=initial_balance,
            window_size=window_size,
            max_holding_time=max_holding_time,
            max_drawdown_threshold=max_drawdown_threshold,
        ),
        n_episodes=1,
        verbose=True,
    )
    print("RandomAgent evaluate result:", eval_random)

    # === BuyAndHoldAgent ===
    env_buyhold = MyTradingEnv(
        df=df,
        initial_balance=initial_balance,
        window_size=window_size,
        max_holding_time=max_holding_time,
        max_drawdown_threshold=max_drawdown_threshold,
    )
    buyhold_agent = BuyAndHoldAgent(initial_buy_step=10)
    steps_buyhold, portfolio_buyhold = run_and_plot_agent(
        buyhold_agent, env_buyhold, "BuyAndHoldAgent"
    )
    buyhold_final = portfolio_buyhold[-1]

    eval_buyhold = buyhold_agent.evaluate(
        env=MyTradingEnv(
            df=df,
            initial_balance=initial_balance,
            window_size=window_size,
            max_holding_time=max_holding_time,
            max_drawdown_threshold=max_drawdown_threshold,
        ),
        n_episodes=1,
        verbose=True,
    )
    print("BuyAndHoldAgent evaluate result:", eval_buyhold)

    # === MovingAverageAgent ===
    env_ma = MyTradingEnv(
        df=df,
        initial_balance=initial_balance,
        window_size=window_size,
        max_holding_time=max_holding_time,
        max_drawdown_threshold=max_drawdown_threshold,
    )
    ma_agent = MovingAverageAgent(
        name="MovingAverageAgent",
        fast_period=10,
        slow_period=30,
        use_exponential=False,
    )
    steps_ma, portfolio_ma = run_and_plot_agent(ma_agent, env_ma, "MovingAverageAgent")
    ma_final = portfolio_ma[-1]

    eval_ma = ma_agent.evaluate(
        env=MyTradingEnv(
            df=df,
            initial_balance=initial_balance,
            window_size=window_size,
            max_holding_time=max_holding_time,
            max_drawdown_threshold=max_drawdown_threshold,
        ),
        n_episodes=1,
        verbose=True,
    )
    print("MovingAverageAgent evaluate result:", eval_ma)

    plt.figure(figsize=(14, 8))
    plt.plot(
        steps_random,
        portfolio_random,
        label=f"RandomAgent (Final: ${random_final:.2f})",
        alpha=0.7,
    )
    plt.plot(
        steps_buyhold,
        portfolio_buyhold,
        label=f"BuyAndHoldAgent (Final: ${buyhold_final:.2f})",
        alpha=0.7,
    )
    plt.plot(
        steps_ma,
        portfolio_ma,
        label=f"MovingAverageAgent (Final: ${ma_final:.2f})",
        alpha=0.7,
    )

    plt.title("Сравнение агентов по стоимости портфеля")
    plt.xlabel("Шаг")
    plt.ylabel("Стоимость портфеля")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
