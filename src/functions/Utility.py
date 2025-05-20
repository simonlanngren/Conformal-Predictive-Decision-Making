"""
utility.py

Provides utility-related helper functions for evaluating predictive decisions.
Includes creation of utility functions, sequences, as well as logic for computing
expected utilities and optimal decision-making.

Authors: Simon Lanngren (simlann@chalmers.se), Martin Toremark (toremark@chalmers.se)
"""

import numpy as np


class Utility:
    """
    Static methods for utility-based evaluation and decision-making.
    """
    
    @staticmethod
    def create_utility_func(threshold, tp, tn, fp, fn):
        """
        Create a binary classification utility function with configurable payoff matrix.

        Parameters
        ----------
        threshold : float
            Threshold above which y is interpreted as class 1.
        tp : float
            Utility for true positive.
        tn : float
            Utility for true negative.
        fp : float
            Utility for false positive.
        fn : float
            Utility for false negative.

        Returns
        -------
        callable
            A function utility(y, decision) that returns the utility of the decision given the true label.
        """
        def utility_func(
            y_value, decision, threshold=threshold, tp=tp, tn=tn, fp=fp, fn=fn
        ):
            y_value = int(y_value >= threshold)
            return (
                tp
                if decision and y_value
                else fp
                if decision
                else fn
                if y_value
                else tn
            )

        return utility_func

    @staticmethod
    def create_utility_sequence(ys, decision, utility):
        """
        Apply a utility function to a sequence of true labels for a fixed decision.

        Parameters
        ----------
        ys : array-like
            Array of y values.
        decision : int
            Decision to evaluate (e.g., 0 or 1).
        utility : callable
            Utility function taking (y, decision) as input.

        Returns
        -------
        np.ndarray
            Array of utility values.
        """
        return np.array([utility(y, decision) for y in ys])

    @staticmethod
    def compute_expected_utility(cdf_vals, utility_func, decision):
        """
        Compute the expected utility based on a predictive distribution.

        Parameters
        ----------
        cdf_vals : array-like
            Sorted predictive distribution values 
            (e.g., CPD or posterior distribution).
        utility_func : callable
            Utility function taking (y, decision) as input.
        decision : int
            Decision to evaluate.

        Returns
        -------
        float
            Expected utility value.
        """
        utilities = np.array([utility_func(y, decision) for y in cdf_vals])

        cdf_len = len(cdf_vals)
        delta_Q_star = np.diff(np.array([i / cdf_len for i in range(cdf_len)]))

        expected_utility = np.sum(utilities[:-1] * delta_Q_star)

        return expected_utility

    @staticmethod
    def optimal_decision_making(Decisions, y_test, utility_func):
        """
        Select the optimal decision for each test instance based on known labels.

        Parameters
        ----------
        Decisions : set
            A set of possible decision values to evaluate.
        y_test : array-like
            Ground truth labels for testing.
        utility_func : callable
            A function that computes the utility given a true label and a decision.

        Returns
        -------
        tuple of (list of int, list of float)
            Optimal decisions and corresponding utility values.
        """
        optimal_decisions, utilities = zip(
            *[
                max(((d, utility_func(y, d)) for d in Decisions), key=lambda x: x[1])
                for y in y_test
            ]
        )

        return optimal_decisions, utilities

    @staticmethod
    def make_decisions(expected_utilities, utility_func, y_test):
        """
        Select the best decision for each test instance based on expected utility.

        Parameters
        ----------
        expected_utilities : list of list of float
            A list where each sublist contains expected utilities for all test instances,
            one sublist per decision value.
        utility_func : callable
            A function that computes the utility given a true label and a decision.
        y_test : array-like
            Ground truth labels for testing.

        Returns
        -------
        tuple of (list of int, list of float)
            Selected decisions and corresponding utility values
        """
        decisions_made = []
        utilities = []
        for i, expected_utility in enumerate(zip(*expected_utilities)):
            max_utility = max(expected_utility)
            decision = expected_utility.index(max_utility)
            decisions_made.append(decision)
            utilities.append(utility_func(y_test[i], decision))

        return decisions_made, utilities
