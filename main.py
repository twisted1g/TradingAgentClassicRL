import pandas as pd
import matplotlib.pyplot as plt

from agents.random_agent import RandomAgent
from agents.buyhold_agent import BuyAndHoldAgent
from agents.moving_average_agent import MovingAverageAgent
from envs.trading_env import MyTradingEnv
from agents.utils import evaluate_agent


def run_and_plot_agent(agent, env, agent_name):
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
    data_path = "data/data_1m.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True, date_format="iso8601")

    df = df.iloc[:100_000]
    print(f"Загружено {len(df)} строк данных.")

    # --- RandomAgent ---
    env_random = MyTradingEnv(
        df=df,
        initial_balance=1000.0,
        window_size=10,
        max_holding_time=10**8,
        max_drawdown_threshold=1,
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

    env_buyhold = MyTradingEnv(
        df=df,
        initial_balance=1000.0,
        window_size=10,
        max_holding_time=10**8,
        max_drawdown_threshold=1,
    )
    buyhold_agent = BuyAndHoldAgent(initial_buy_step=10)
    steps_buyhold, portfolio_buyhold = run_and_plot_agent(
        buyhold_agent, env_buyhold, "BuyAndHoldAgent"
    )
    buyhold_final = portfolio_buyhold[-1]

    env_ma = MyTradingEnv(
        df=df,
        initial_balance=1000.0,
        window_size=10,
        max_holding_time=10**8,
        max_drawdown_threshold=1,
    )
    ma_agent = MovingAverageAgent(
        name="MovingAverageAgent",
        fast_period=10,
        slow_period=30,
        use_exponential=False,
    )
    steps_ma, portfolio_ma = run_and_plot_agent(ma_agent, env_ma, "MovingAverageAgent")
    ma_final = portfolio_ma[-1]

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
