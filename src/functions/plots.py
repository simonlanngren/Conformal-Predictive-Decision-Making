import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Plots:
    @staticmethod
    def plot_average_utility(experiment, mode, confidence=0.95):
        # Check number of runs
        first_key = next(iter(experiment))
        n_runs = len(experiment[first_key]) # TODO Error handling
        if n_runs == 1:
            for i, (method, runs) in enumerate(experiment.items()):
                utilities = runs[0]
                n_samples = len(utilities)
                x = list(range(1, n_samples + 1))
                cumulative_avg = np.cumsum(utilities) / np.arange(1, n_samples + 1)
                plt.plot(x, cumulative_avg, label=method, alpha=1-0.05*i)

            plt.xlabel('Test Case')
            plt.ylabel('Average Utility')
            plt.legend()
            plt.title(f"Average Utility Over Test Cases - {mode} Setting")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            data = []
            for method, runs in experiment.items():
                for utilities in runs:
                    cumulative_avg = np.cumsum(utilities) / np.arange(1, len(utilities) + 1)
                    for i, utility in enumerate(cumulative_avg):
                        data.append({
                            "test_case": i + 1,
                            "method": method,
                            "utility": utility
                        })
            df = pd.DataFrame(data)

            for method in df["method"].unique():
                method_df = df[df["method"] == method]
                if method == "Optimal":
                    sns.lineplot(
                        data=method_df,
                        x="test_case",
                        y="utility",
                        label=method,
                        errorbar=None,
                        linewidth=2.5,
                        linestyle="--"
                    )
                else:
                    sns.lineplot(
                        data=method_df,
                        x="test_case",
                        y="utility",
                        label=method,
                        errorbar=('ci', int(confidence * 100))
                    )

            plt.xlabel("Test Case")
            plt.ylabel("Cumulative Average Utility")
            plt.title(f"Average Utility with Confidence Intervals - {mode} Setting")
            plt.legend(title="Method")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
    @staticmethod
    def plot_cumulative_regret(experiment, mode, confidence=0.95):
        # Check number of runs
        first_key = next(iter(experiment))
        n_runs = len(experiment[first_key]) # TODO Error handling
        if n_runs == 1:
            optimal_utilities = np.array(experiment["Optimal"][0])
            x = np.arange(1, len(optimal_utilities) + 1)
            for method, runs in experiment.items():
                if method == "Optimal":
                    continue
                
                utilities = np.array(runs[0])
                regret = optimal_utilities - utilities
                cumulative_regret = np.cumsum(regret)
                plt.plot(x, cumulative_regret, label=method)
                
            plt.xlabel("Test Case")
            plt.ylabel("Cumulative Regret")
            plt.title(f"Cumulative Regret - {mode} Setting")
            plt.legend(title="Method")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            data = []
            for method, runs in experiment.items():
                if method == "Optimal":
                    continue
                
                for i, run in enumerate(runs):
                    method_utilities = np.array(run)
                    optimal_utilities = np.array(experiment["Optimal"][i])
                    regrets = optimal_utilities - method_utilities
                    cumulative_regrets = np.cumsum(regrets)

                    for i, regret in enumerate(cumulative_regrets):
                        data.append({
                            "test_case": i + 1,
                            "method": method,
                            "regret": regret
                        })
            df = pd.DataFrame(data)

            for method in df["method"].unique():
                method_df = df[df["method"] == method]
                sns.lineplot(
                    data=method_df,
                    x="test_case",
                    y="regret",
                    label=method,
                    errorbar=('ci', int(confidence * 100))
                )

            plt.xlabel("Test Case")
            plt.ylabel("Cumulative Regret")
            plt.title(f"Cumulative Regret with {int(confidence * 100)}% Confidence - {mode} Setting")
            plt.legend(title="Method")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_cpd(cpd_vals):
        sorted_vals = np.sort(cpd_vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        plt.plot(sorted_vals, cdf, marker='.', linestyle='-')
        plt.title("Cumulative Plot (CPD)")
        plt.xlabel("Value")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.show()