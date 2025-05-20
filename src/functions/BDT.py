"""
bdt.py

Implements Bayesian Decision Theory-based prediction strategies for both
inductive and online settings. Contains the BDT class with static methods
for computing utility-based decisions using a given model.

Authors: Simon Lanngren (simlann@chalmers.se), Martin Toremark (toremark@chalmers.se)
"""

import numpy as np
from sklearn.base import clone
from .utility import Utility


class BDT:
    """
    Provides static methods for utility-optimized decision-making using using
    bayesian posterior distributions in both inductive and online settings.
    """

    @staticmethod
    def inductive(model, Decisions, utility_func, X_train, y_train, X_test, y_test):
        """
        Inductive version of Bayesian Decision Theory.

        Trains the model on the entire training set and makes predictions on the test set.
        Calculates expected utilities for each decision and returns the utility outcomes.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            Scikit-learn compatible model.
        Decisions : set
            A set of possible decision values to evaluate.
        utility_func : callable
            A function that computes the utility given a prediction and a decision.
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
        preds = model.predict(X_test)

        expected_utilities = []
        for d in Decisions:
            expected_utilities_d = Utility.create_utility_sequence(
                preds, d, utility_func
            )
            expected_utilities.append(expected_utilities_d)

        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)

        return utilities

    @staticmethod
    def online(model, Decisions, utility_func, X_train, y_train, X_test, y_test):
        """
        Online version of Bayesian Decision Theory.

        Iteratively fits the model on the cumulative seen data and makes one prediction at a time.
        Expected utilities are calculated per decision, based on sequential predictions.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            Scikit-learn compatible model.
        Decisions : set
            A set of possible decision values to evaluate.
        utility_func : callable
            A function that computes the utility given a prediction and a decision.
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
        X_seen = X_train
        y_seen = y_train

        expected_utilities = []
        preds = []
        for i, (x, y) in enumerate(zip(X_test, y_test)):
            model_i = clone(model)
            model_i.fit(X_seen, y_seen)
            pred = model_i.predict(X_test[i].reshape(1, -1))

            preds.append(pred)

            X_seen = np.append(X_seen, [x], axis=0)
            y_seen = np.append(y_seen, [y])

        for d in Decisions:
            expected_utility_d = Utility.create_utility_sequence(preds, d, utility_func)
            expected_utilities.append(expected_utility_d)

        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)

        return utilities
