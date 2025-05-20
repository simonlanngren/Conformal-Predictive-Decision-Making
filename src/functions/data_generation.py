"""
data_generation.py

Generates synthetic regression datasets using scikit-learn's built-in generators,
with post-processing to fit the specific decision scenario. Used for benchmarking
the predictive decision making systems.

Authors: Simon Lanngren (simlann@chalmers.se), Martin Toremark (toremark@chalmers.se)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_regression,
    make_friedman1,
    make_friedman2,
    make_friedman3,
)
from sklearn.preprocessing import MinMaxScaler


class DataGeneration:
    """
    A collection of static methods for generating and post-processing
    synthetic regression datasets.
    """

    @staticmethod
    def generate_data(
        n_samples=10000,
        n_features=20,
        relationship="make_regression",
        noise=0.1,
        random_state=None,
    ):
        """
        Generate a synthetic regression dataset using a specified relationship type.

        Supported types include:
        - 'make_regression'
        - 'friedman1'
        - 'friedman2'
        - 'friedman3'

        The target is min-max scaled to the [0, 1] range.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate (default is 10,000).
        n_features : int, optional
            Number of features (only applicable to some relationships, default is 20).
        relationship : str, optional
            Type of synthetic relationship to use (default is 'make_regression').
            Must be one of {'make_regression', 'friedman1', 'friedman2', 'friedman3'}.
        noise : float, optional
            Noise level to add to the data (default is 0.1).
        random_state : int or None, optional
            Seed for reproducibility.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with features as columns `Feature_0` to `Feature_{n-1}`
            and a min-max scaled `Target` column.
        """
        if relationship == "make_regression":
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_features,
                noise=noise,
                random_state=random_state,
            )

        elif relationship == "friedman1":
            X, y = make_friedman1(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise,
                random_state=random_state,
            )

        elif relationship == "friedman2":
            X, y = make_friedman2(
                n_samples=n_samples, noise=noise, random_state=random_state
            )

        elif relationship == "friedman3":
            X, y = make_friedman3(
                n_samples=n_samples, noise=noise, random_state=random_state
            )

        else:
            raise ValueError(f"Unsupported relationship: {relationship}")

        # Min-max scale the target
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
        df["Target"] = y_scaled

        return df

    @staticmethod
    def add_epsilon(df):
        """
        Add machine epsilon-level noise to the target column to avoid duplicate values.

        This is necessary because we use bootstrap sampling and online conformal prediction
        methods that require all target values to be unique. Adding small perturbations
        ensures numerical stability and prevents errors caused by duplicate values.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame with a column named 'Target'.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame where the 'Target' column has been slightly
            perturbed using scaled machine epsilon noise.
        """
        MACHINE_EPSILON = lambda x: np.abs(x) * np.finfo(np.float64).eps

        target_array = df["Target"].to_numpy(dtype=np.float64)
        noise = (
            np.random.uniform(1, 10, size=len(df)) * MACHINE_EPSILON(target_array) * 1e5
        )
        adjusted_y = target_array + noise

        df_copy = df.copy()
        df_copy["Target"] = adjusted_y

        return df_copy
