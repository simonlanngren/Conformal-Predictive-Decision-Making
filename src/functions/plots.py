"""
plots.py

General visualization utilities for predictive modeling and decision-making.
Includes tools for evaluating experiments (e.g., average utility, cumulative regret),
as well as inspecting prediction distributions and dataset characteristics.

Authors: Simon Lanngren (simlann@chalmers.se), Martin Toremark (toremark@chalmers.se)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class Plots:
    """
    Static plotting utilities for analyzing predictive models and datasets.
    Supports evaluation plots (e.g., regret, utility) and distributional analysis
    (e.g., CPDs, target distributions).
    """

    @staticmethod
    def plot_average_utility(
        experiment, output_folder, output_filename, confidence=0.95
    ):
        """
        Plot average utility across test cases for each method.

        If multiple runs are provided per method, confidence intervals are shown.
        Saves the plot as a PDF.

        Parameters
        ----------
        experiment : dict
            A dictionary mapping method names to lists of utility sequences.
            Each value should be a list of lists with one utility list per run.
        output_folder : str
            Directory where the figure should be saved.
        output_filename : str
            Name of the PDF file to save.
        confidence : float, optional
            Confidence level for error bars when plotting multiple runs (default is 0.95).
        """

        first_key = next(iter(experiment))
        n_runs = len(experiment[first_key])
        sns.set_palette("colorblind")
        if n_runs == 1:
            for i, (method, runs) in enumerate(experiment.items()):
                utilities = runs[0]
                n_samples = len(utilities)
                x = list(range(1, n_samples + 1))
                cumulative_avg = np.cumsum(utilities) / np.arange(1, n_samples + 1)
                plt.plot(x, cumulative_avg, label=method, alpha=1 - 0.05 * i)

            plt.xlabel("Test Case")
            plt.ylabel("Average Utility")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(output_folder, output_filename)
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            plt.close()
        else:
            data = []
            for method, runs in experiment.items():
                for utilities in runs:
                    cumulative_avg = np.cumsum(utilities) / np.arange(
                        1, len(utilities) + 1
                    )
                    for i, utility in enumerate(cumulative_avg):
                        data.append(
                            {"test_case": i + 1, "method": method, "utility": utility}
                        )
            df = pd.DataFrame(data)

            for method in df["method"].unique():
                method_df = df[df["method"] == method]
                if method == "Optimal":
                    continue
                else:
                    sns.lineplot(
                        data=method_df,
                        x="test_case",
                        y="utility",
                        label=method,
                        errorbar=("ci", int(confidence * 100)),
                    )

            plt.xlabel("Test Case")
            plt.ylabel("Average Utility")
            plt.legend(title="Method")
            plt.grid(True)
            plt.tight_layout()

            save_path = os.path.join(output_folder, output_filename)
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            plt.close()

    @staticmethod
    def plot_cumulative_regret(
        experiment, output_folder, output_filename, confidence=0.95
    ):
        """
        Plot cumulative regret relative to the 'Optimal' decision for each method.

        If multiple runs are provided per method, confidence intervals are shown.
        Saves the plot as a PDF.

        Parameters
        ----------
        experiment : dict
            A dictionary mapping method names to lists of utility sequences.
            Must include the key 'Optimal' as a reference baseline.
        output_folder : str
            Directory where the figure should be saved.
        output_filename : str
            Name of the PDF file to save.
        confidence : float, optional
            Confidence level for error bars when plotting multiple runs (default is 0.95).
        """

        first_key = next(iter(experiment))
        n_runs = len(experiment[first_key])
        sns.set_palette("colorblind")
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
            plt.legend(title="Method")
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(output_folder, output_filename)
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            plt.close()
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
                        data.append(
                            {"test_case": i + 1, "method": method, "regret": regret}
                        )
            df = pd.DataFrame(data)

            for method in df["method"].unique():
                method_df = df[df["method"] == method]
                sns.lineplot(
                    data=method_df,
                    x="test_case",
                    y="regret",
                    label=method,
                    errorbar=("ci", int(confidence * 100)),
                )

            plt.xlabel("Test Case")
            plt.ylabel("Cumulative Regret")
            plt.legend(title="Method")
            plt.grid(True)
            plt.tight_layout()

            save_path = os.path.join(output_folder, output_filename)
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            plt.close()

    @staticmethod
    def plot_cpd(cpd_vals):
        """
        Plot the cumulative distribution of a conformal predictive distribution (CPD).

        Parameters
        ----------
        cpd_vals : array-like
            Sorted or unsorted values from the predictive distribution.
        """

        sorted_vals = np.sort(cpd_vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_vals, cdf, marker=".", linestyle="-")
        plt.title("Cumulative Plot (CPD)")
        plt.xlabel("Value")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_target_distribution(
        df,
        figsize=(6, 3.5),
        bins=30,
        kde=True,
        store_target_plot=False,
        output_folder=None,
        output_filename=None,
    ):
        """
        Plot the distribution of the 'Target' column in a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing a column named 'Target'.
        figsize : tuple, optional
            Figure size in inches (default is (6, 3.5)).
        bins : int, optional
            Number of histogram bins (default is 30).
        kde : bool, optional
            Whether to overlay a KDE (default is True).
        store_target_plot : bool, optional
            Whether to save the plot instead of displaying it (default is False).
        output_folder : str or None, optional
            Directory where the figure should be saved if `store_target_plot=True`.
        output_filename : str or None, optional
            Name of the PDF file to save if `store_target_plot=True`.
        """
        plt.figure(figsize=figsize)
        sns.histplot(data=df, x="Target", bins=bins, kde=kde, edgecolor="black")
        plt.xlabel("y")
        plt.ylabel("Frequency")
        plt.tight_layout()
        if store_target_plot:
            save_path = os.path.join(output_folder, output_filename)
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_first_feature_distribution(df, figsize=(6, 3.5), bins=30, kde=True):
        """
        Plot the distribution of the first feature (Feature_0) in a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing a column named 'Feature_0'.
        figsize : tuple, optional
            Figure size in inches (default is (6, 3.5)).
        bins : int, optional
            Number of histogram bins (default is 30).
        kde : bool, optional
            Whether to overlay a KDE (default is True).
        """
        plt.figure(figsize=figsize)
        sns.histplot(data=df, x="Target", bins=bins, kde=kde, edgecolor="black")
        plt.title("Feature_0 Distribution")
        plt.xlabel("Feature_0")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
