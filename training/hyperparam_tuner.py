from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
from itertools import product
from .dataclasses import TrainingConfig
from .training_manager import TrainingManager


class HyperparameterTuner:
    def __init__(self, training_manager: TrainingManager):
        self.training_manager = training_manager
        self.results: List[Dict] = []

    def grid_search(
        self,
        agent_class,
        env,
        param_grid: Dict[str, List],
        base_config: Dict,
        n_episodes: int = 1000,
        metric: str = "mean_reward",
        verbose: bool = True,
    ) -> pd.DataFrame:
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        total_combinations = np.prod([len(v) for v in param_values])

        if verbose:
            print(f"\n{'='*70}")
            print(f"Grid Search: {agent_class.__name__}")
            print(f"Параметры: {param_names}")
            print(f"Всего комбинаций: {total_combinations}")
            print(f"{'='*70}\n")

        results = []

        for idx, params in enumerate(product(*param_values), 1):
            param_dict = dict(zip(param_names, params))

            if verbose:
                print(f"\n[{idx}/{total_combinations}] Тестируем: {param_dict}")

            agent_params = {**base_config, **param_dict}
            agent = agent_class(**agent_params)

            config = TrainingConfig(
                agent_name=f"{agent_class.__name__}_gs{idx}",
                agent_type=agent_class.__name__,
                n_episodes=n_episodes,
                max_steps=base_config.get("max_steps", 1000),
                **agent_params,
            )

            experiment_name = f"grid_search_{agent_class.__name__}_{idx}"
            train_results = self.training_manager.train_agent(
                agent, env, config, experiment_name=experiment_name, verbose=False
            )

            eval_results = agent.evaluate(env, n_episodes=20)

            result = {
                **param_dict,
                "experiment_name": experiment_name,
                "mean_reward": eval_results["mean_reward"],
                "std_reward": eval_results["std_reward"],
                "final_portfolio": train_results["final_metrics"]["portfolio_value"],
                "q_table_size": len(agent.q_table),
                "training_time": train_results["training_time"],
            }
            results.append(result)

            if verbose:
                print(f"  → {metric}: {result[metric]:.2f}")

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(metric, ascending=False)

        results_dir = self.training_manager.base_log_dir / "hyperparameter_search"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"{agent_class.__name__}_{timestamp}.csv"
        df_results.to_csv(results_path, index=False)

        if verbose:
            print(f"\n{'='*70}")
            print("Топ-5 конфигураций:")
            print(df_results.head(5).to_string())
            print(f"\nРезультаты сохранены: {results_path}")
            print(f"{'='*70}\n")

        self.results.append(
            {
                "agent_class": agent_class.__name__,
                "param_grid": param_grid,
                "results": df_results,
                "best_params": df_results.iloc[0].to_dict(),
            }
        )

        return df_results

    def random_search(
        self,
        agent_class,
        env,
        param_distributions: Dict[str, tuple],
        base_config: Dict,
        n_trials: int = 20,
        n_episodes: int = 1000,
        metric: str = "mean_reward",
        verbose: bool = True,
    ) -> pd.DataFrame:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Random Search: {agent_class.__name__}")
            print(f"Количество проб: {n_trials}")
            print(f"{'='*70}\n")

        results = []

        for trial in range(n_trials):
            param_dict = {}
            for param_name, (min_val, max_val, scale) in param_distributions.items():
                if scale == "log":
                    param_dict[param_name] = 10 ** np.random.uniform(
                        np.log10(min_val), np.log10(max_val)
                    )
                else:
                    param_dict[param_name] = np.random.uniform(min_val, max_val)

            if verbose:
                print(f"\n[{trial+1}/{n_trials}] Тестируем: {param_dict}")

            agent_params = {**base_config, **param_dict}
            agent = agent_class(**agent_params)

            config = TrainingConfig(
                agent_name=f"{agent_class.__name__}_rs{trial+1}",
                agent_type=agent_class.__name__,
                n_episodes=n_episodes,
                max_steps=base_config.get("max_steps", 1000),
                **agent_params,
            )

            experiment_name = f"random_search_{agent_class.__name__}_{trial+1}"
            train_results = self.training_manager.train_agent(
                agent, env, config, experiment_name=experiment_name, verbose=False
            )

            eval_results = agent.evaluate(env, n_episodes=20)

            result = {
                **param_dict,
                "experiment_name": experiment_name,
                "mean_reward": eval_results["mean_reward"],
                "std_reward": eval_results["std_reward"],
                "final_portfolio": train_results["final_metrics"]["portfolio_value"],
                "q_table_size": len(agent.q_table),
                "training_time": train_results["training_time"],
            }
            results.append(result)

            if verbose:
                print(f"  → {metric}: {result[metric]:.2f}")

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(metric, ascending=False)

        results_dir = self.training_manager.base_log_dir / "hyperparameter_search"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"{agent_class.__name__}_random_{timestamp}.csv"
        df_results.to_csv(results_path, index=False)

        if verbose:
            print(f"\n{'='*70}")
            print("Топ-5 конфигураций:")
            print(df_results.head(5).to_string())
            print(f"\nРезультаты сохранены: {results_path}")
            print(f"{'='*70}\n")

        return df_results
