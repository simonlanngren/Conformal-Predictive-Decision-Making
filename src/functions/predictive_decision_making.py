"""
predictive_decision_making.py

Implements a class for binary decision-making using point predictive models.
Supports both inductive and online learning settings.

Authors: Simon Lanngren (simlann@chalmers.se), Martin Toremark (toremark@chalmers.se)
"""

import numpy as np
from sklearn.base import clone
from src.functions.model_selection import ModelSelection


class PredictiveBinaryDecisionMaking:
    """
    Binary point predictive decision-making implementation.

    Supports both inductive and online settings, where decisions are made
    by comparing model predictions against a threshold and computing utility.
    """

    @staticmethod
    def inductive(model, utility_func, threshold, X_train, y_train, X_test, y_test):
        """
        Make binary decisions in an inductive setting using a trained model.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            Scikit-learn compatible model.
        utility_func : callable
            A function that computes the utility given a prediction and a decision.
        threshold : float
            Decision threshold for classifying outputs as 1 or 0.
        X_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray
            Ground truth labels for training.
        X_test : np.ndarray
            Feature matrix for testing.
        y_test : np.ndarray
            Ground truth labels for testing.

        Returns
        -------
        list of float
            Utility values for each decision made.
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        decisions_made = np.where(y_pred >= threshold, 1, 0)
        utilities = [utility_func(y, d) for y, d in zip(y_test, decisions_made)]
        return utilities

    @staticmethod
    def online(
        model,
        utility_func,
        threshold,
        X_train,
        y_train,
        X_test,
        y_test,
        param_grid,
        n_splits=5,
        random_state=None,
    ):
        """
        Make binary decisions in an online setting with dynamic model tuning.

        At each step, the model is re-tuned and retrained using all seen data.
        The new model is used to predict and classify the next test point.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            Scikit-learn compatible model.
        utility_func : callable
            A function that computes the utility given a prediction and a decision.
        threshold : float
            Decision threshold for classifying outputs as 1 or 0.
        X_train : np.ndarray
            Feature matrix for initial training.
        y_train : np.ndarray
            Ground truth labels for initial training.
        X_test : np.ndarray
            Feature matrix for testing (streamed sequentially).
        y_test : np.ndarray
            Ground truth labels for testing.
        param_grid : dict
            Hyperparameter grid for model selection.
        n_splits : int, optional
            Number of cross-validation splits for model tuning (default is 5).
        random_state : int or None, optional
            Seed for reproducibility (default is None).

        Returns
        -------
        list of float
            Utility values for each decision made.
        """
        X_seen = X_train
        y_seen = y_train

        decisions_made = []
        for x, y in zip(X_test, y_test):
            model_i = clone(model)

            best_params, _ = ModelSelection.model_selection(
                X_seen,
                y_seen,
                model_i,
                param_grid,
                n_splits=n_splits,
                random_state=random_state,
            )

            model_i = clone(model)
            model_i.set_params(**best_params)
            model_i.fit(X_seen, y_seen)

            y_pred = model_i.predict(x.reshape(1, -1))
            decisions_made.append(np.where(y_pred >= threshold, 1, 0))

            X_seen = np.append(X_seen, [x], axis=0)
            y_seen = np.append(y_seen, [y])

        utilities = [utility_func(y, d) for y, d in zip(y_test, decisions_made)]

        return utilities
