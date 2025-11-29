import pandas as pd
from pathlib import Path


from envs.trading_env import MyTradingEnv
from agents.classical.qlearning_agent import QLearningAgent
from agents.classical.sarsa_lambda_agent import SarsaLambdaAgent
from training import TrainingManager, TrainingConfig, HyperparameterTuner

from analyze import ResultsAnalyzer, HyperparameterAnalyzer


def example_1_simple_training():
    df = pd.read_csv("data/data_1h_2023.csv")
    df["date"] = pd.to_datetime(df["date_close"])

    env = MyTradingEnv(
        df=df,
        initial_balance=1000.0,
        window_size=10,
        commission=0.0001,
        slippage=0.0005,
    )

    agent = QLearningAgent(
        n_actions=3,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )

    config = TrainingConfig(
        agent_name="QLearning_v1",
        agent_type="QLearning",
        n_episodes=2000,
        max_steps=50,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        eval_frequency=100,
        save_frequency=500,
    )

    manager = TrainingManager(
        base_log_dir="training_data/logs",
        base_checkpoint_dir="training_data/checkpoints",
    )
    results = manager.train_agent(
        agent=agent,
        env=env,
        config=config,
        experiment_name="qlearning_baseline",
        verbose=True,
    )

    print(f"\nРезультаты обучения:")
    print(f"Эксперимент: {results['experiment_name']}")
    print(f"Логи: {results['log_dir']}")
    print(f"Чекпоинты: {results['checkpoint_dir']}")
    print(f"Финальный агент: {results['final_agent_path']}")


def example_2_compare_algorithms():
    df = pd.read_csv("data/data_1h_2023.csv")
    df["date"] = pd.to_datetime(df["date_close"])

    env = MyTradingEnv(df=df, initial_balance=1000.0)
    manager = TrainingManager()

    print("\n=== Обучение Q-Learning ===")
    qlearning_agent = QLearningAgent(
        learning_rate=0.1, discount_factor=0.99, epsilon_decay=0.995
    )
    qlearning_config = TrainingConfig(
        agent_name="QLearning",
        agent_type="QLearning",
        n_episodes=200,
        max_steps=1000,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )
    qlearning_results = manager.train_agent(
        qlearning_agent, env, qlearning_config, experiment_name="comparison_qlearning"
    )

    print("\n=== Обучение SARSA(λ) ===")
    sarsa_agent = SarsaLambdaAgent(
        learning_rate=0.1, discount_factor=0.99, epsilon_decay=0.995, lambda_param=0.9
    )
    sarsa_config = TrainingConfig(
        agent_name="SARSA_lambda",
        agent_type="SarsaLambda",
        n_episodes=200,
        max_steps=1000,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        lambda_param=0.9,
    )
    sarsa_results = manager.train_agent(
        sarsa_agent, env, sarsa_config, experiment_name="comparison_sarsa_lambda"
    )

    analyzer = ResultsAnalyzer()

    analyzer.compare_experiments(
        experiment_names=["comparison_qlearning", "comparison_sarsa_lambda"],
        save_path="results/comparison_plots.png",
    )

    summary_table = analyzer.create_summary_table(
        ["comparison_qlearning", "comparison_sarsa_lambda"]
    )
    print("\n=== Сводная таблица ===")
    print(summary_table.to_string())
    summary_table.to_csv("results/comparison_summary.csv", index=False)


def example_3_grid_search():
    df = pd.read_csv("data/data_1h_2023.csv")
    df["date"] = pd.to_datetime(df["date_close"])

    env = MyTradingEnv(df=df, initial_balance=1000.0)
    manager = TrainingManager()
    tuner = HyperparameterTuner(manager)

    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "discount_factor": [0.95, 0.99, 0.999],
        "epsilon_decay": [0.99, 0.995, 0.999],
    }

    base_config = {
        "n_actions": 3,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "max_steps": 1000,
    }

    print("\n=== Grid Search для Q-Learning ===")
    results_df = tuner.grid_search(
        agent_class=QLearningAgent,
        env=env,
        param_grid=param_grid,
        base_config=base_config,
        n_episodes=1000,
        metric="mean_reward",
        verbose=True,
    )

    print("\n=== Топ-5 конфигураций ===")
    print(results_df.head(5).to_string())

    # Анализ результатов
    hp_analyzer = HyperparameterAnalyzer()

    # Влияние learning_rate
    hp_analyzer.plot_hyperparameter_impact(
        results_df,
        param_name="learning_rate",
        metric="mean_reward",
        save_path="results/lr_impact.png",
    )

    # Heatmap для двух параметров
    hp_analyzer.create_heatmap(
        results_df,
        param1="learning_rate",
        param2="discount_factor",
        metric="mean_reward",
        save_path="results/lr_gamma_heatmap.png",
    )


# ============================================================================
# ПРИМЕР 4: Random Search
# ============================================================================


def example_4_random_search():
    """Random search для более широкого исследования"""

    df = pd.read_csv("data/data_1h_2023.csv")
    df["date"] = pd.to_datetime(df["date"])

    env = MyTradingEnv(df=df, initial_balance=1000.0)
    manager = TrainingManager()
    tuner = HyperparameterTuner(manager)

    # Определяем распределения параметров
    param_distributions = {
        "learning_rate": (0.001, 0.5, "log"),  # логарифмическая шкала
        "discount_factor": (0.90, 0.999, "linear"),
        "epsilon_decay": (0.98, 0.9999, "linear"),
    }

    base_config = {
        "n_actions": 3,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "max_steps": 1000,
    }

    print("\n=== Random Search для Q-Learning ===")
    results_df = tuner.random_search(
        agent_class=QLearningAgent,
        env=env,
        param_distributions=param_distributions,
        base_config=base_config,
        n_trials=30,
        n_episodes=1000,
        metric="mean_reward",
        verbose=True,
    )

    print("\n=== Топ-5 конфигураций ===")
    print(results_df.head(5).to_string())

    # Сохраняем лучшие параметры
    best_params = results_df.iloc[0].to_dict()
    print(f"\nЛучшие параметры:")
    for key, value in best_params.items():
        if key not in ["experiment_name", "mean_reward", "std_reward"]:
            print(f"  {key}: {value}")


# ============================================================================
# ПРИМЕР 5: Анализ существующих экспериментов
# ============================================================================


def example_5_analyze_experiments():
    """Анализ ранее запущенных экспериментов"""

    analyzer = ResultsAnalyzer(log_base_dir="logs")

    # Список всех экспериментов
    experiments = analyzer.list_experiments()
    print(f"\nНайдено экспериментов: {len(experiments)}")
    for exp in experiments:
        print(f"  - {exp}")

    # Детальный анализ конкретного эксперимента
    if experiments:
        exp_name = experiments[0]
        print(f"\n=== Анализ эксперимента: {exp_name} ===")

        # Информация
        analyzer.print_experiment_info(exp_name)

        # Детальные графики
        analyzer.plot_single_experiment(
            exp_name, save_path=f"results/{exp_name}_detailed.png"
        )

    # Сравнение нескольких экспериментов
    if len(experiments) >= 2:
        print("\n=== Сравнение экспериментов ===")
        analyzer.compare_experiments(
            experiment_names=experiments[:3],  # Первые 3
            save_path="results/multi_experiment_comparison.png",
        )

        # Сводная таблица
        summary = analyzer.create_summary_table(experiments[:3])
        print("\n=== Сводная таблица ===")
        print(summary.to_string())


# ============================================================================
# ПРИМЕР 6: Обучение с кастомными параметрами окружения
# ============================================================================


def example_6_custom_env_params():
    """Обучение с разными параметрами окружения"""

    df = pd.read_csv("data/data_1h_2023.csv")
    df["date"] = pd.to_datetime(df["date"])

    manager = TrainingManager()

    # Вариант 1: Низкие комиссии
    print("\n=== Обучение с низкими комиссиями ===")
    env_low_commission = MyTradingEnv(
        df=df, initial_balance=1000.0, commission=0.00005, slippage=0.0002
    )

    agent1 = QLearningAgent(learning_rate=0.1, discount_factor=0.99)
    config1 = TrainingConfig(
        agent_name="QLearning_low_commission",
        agent_type="QLearning",
        n_episodes=1500,
        max_steps=1000,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        commission=0.00005,
    )

    results1 = manager.train_agent(
        agent1, env_low_commission, config1, experiment_name="env_test_low_commission"
    )

    # Вариант 2: Высокие комиссии
    print("\n=== Обучение с высокими комиссиями ===")
    env_high_commission = MyTradingEnv(
        df=df, initial_balance=1000.0, commission=0.0005, slippage=0.001
    )

    agent2 = QLearningAgent(learning_rate=0.1, discount_factor=0.99)
    config2 = TrainingConfig(
        agent_name="QLearning_high_commission",
        agent_type="QLearning",
        n_episodes=1500,
        max_steps=1000,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        commission=0.0005,
    )

    results2 = manager.train_agent(
        agent2, env_high_commission, config2, experiment_name="env_test_high_commission"
    )

    # Сравнение
    analyzer = ResultsAnalyzer()
    analyzer.compare_experiments(
        ["env_test_low_commission", "env_test_high_commission"],
        save_path="results/commission_comparison.png",
    )


# ============================================================================
# ПРИМЕР 7: Загрузка и дообучение агента
# ============================================================================


def example_7_continue_training():
    """Загрузка сохраненного агента и продолжение обучения"""

    df = pd.read_csv("data/data_1h_2023.csv")
    df["date"] = pd.to_datetime(df["date"])

    env = MyTradingEnv(df=df, initial_balance=1000.0)

    # Загружаем ранее обученного агента
    agent = QLearningAgent()
    agent.load("checkpoints/qlearning_baseline/final_agent.pkl")

    print(f"Загружен агент:")
    print(f"  Эпизодов: {agent.episode_count}")
    print(f"  Q-table size: {len(agent.q_table)}")
    print(f"  Epsilon: {agent.epsilon}")

    # Продолжаем обучение
    manager = TrainingManager()
    config = TrainingConfig(
        agent_name="QLearning_continued",
        agent_type="QLearning",
        n_episodes=1000,  # Еще 1000 эпизодов
        max_steps=1000,
        learning_rate=agent.learning_rate,
        discount_factor=agent.discount_factor,
        epsilon_start=agent.epsilon,  # Начинаем с текущего epsilon
        epsilon_end=0.001,  # Еще больше снижаем
        epsilon_decay=0.999,
    )

    results = manager.train_agent(
        agent, env, config, experiment_name="qlearning_continued_training", verbose=True
    )


# ============================================================================
# ПРИМЕР 8: Batch обучение нескольких агентов
# ============================================================================


def example_8_batch_training():
    """Обучение нескольких конфигураций последовательно"""

    df = pd.read_csv("data/data_1h_2023.csv")
    df["date"] = pd.to_datetime(df["date"])

    env = MyTradingEnv(df=df, initial_balance=1000.0)
    manager = TrainingManager()

    # Определяем несколько конфигураций
    configurations = [
        {
            "name": "qlearning_aggressive",
            "agent": QLearningAgent,
            "lr": 0.2,
            "gamma": 0.95,
            "epsilon_decay": 0.99,
        },
        {
            "name": "qlearning_conservative",
            "agent": QLearningAgent,
            "lr": 0.05,
            "gamma": 0.999,
            "epsilon_decay": 0.999,
        },
        {
            "name": "sarsa_lambda_balanced",
            "agent": SarsaLambdaAgent,
            "lr": 0.1,
            "gamma": 0.99,
            "epsilon_decay": 0.995,
            "lambda_param": 0.9,
        },
    ]

    results_list = []

    for idx, cfg in enumerate(configurations, 1):
        print(f"\n{'='*70}")
        print(f"Обучение {idx}/{len(configurations)}: {cfg['name']}")
        print(f"{'='*70}")

        # Создаем агента
        agent_params = {
            "learning_rate": cfg["lr"],
            "discount_factor": cfg["gamma"],
            "epsilon_decay": cfg["epsilon_decay"],
        }
        if "lambda_param" in cfg:
            agent_params["lambda_param"] = cfg["lambda_param"]

        agent = cfg["agent"](**agent_params)

        # Конфиг обучения
        config = TrainingConfig(
            agent_name=cfg["name"],
            agent_type=cfg["agent"].__name__,
            n_episodes=1500,
            max_steps=1000,
            **agent_params,
        )

        # Обучаем
        results = manager.train_agent(
            agent, env, config, experiment_name=f"batch_{cfg['name']}", verbose=True
        )

        results_list.append({"name": cfg["name"], "results": results})

    # Сравниваем все результаты
    print("\n{'='*70}")
    print("Сравнение всех конфигураций")
    print(f"{'='*70}")

    analyzer = ResultsAnalyzer()
    experiment_names = [f"batch_{cfg['name']}" for cfg in configurations]

    analyzer.compare_experiments(
        experiment_names, save_path="results/batch_comparison.png"
    )

    summary = analyzer.create_summary_table(experiment_names)
    print("\n=== Итоговая таблица ===")
    print(summary.to_string())
    summary.to_csv("results/batch_summary.csv", index=False)


if __name__ == "__main__":
    Path("training_data/results").mkdir(exist_ok=True)
    Path("training_data/logs").mkdir(exist_ok=True)
    Path("training_data/checkpoints").mkdir(exist_ok=True)

    # example_1_simple_training()
    example_2_compare_algorithms()
    # example_3_grid_search()
    # example_4_random_search()
    # example_5_analyze_experiments()
    # example_6_custom_env_params()
    # example_7_continue_training()
    # example_8_batch_training()
