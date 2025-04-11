import numpy as np


class Utility:
    @staticmethod
    def create_utility_func(threshold, tp, tn, fp, fn):
        def utility_func(y_value, decision, threshold=threshold, tp=tp, tn=tn, fp=fp, fn=fn):
            y_value = int(y_value >= threshold)
            return (
                tp if decision and y_value else
                fp if decision else
                fn if y_value else
                tn
            )
                        
        return utility_func
    
    @staticmethod
    def create_utility_sequence(ys, decision, utility):
        return np.array([utility(y, decision) for y in ys])
    
    @staticmethod
    def compute_expected_utility(cdf_vals, utility_func, decision):
        """
        Computes the expected utility using the CDF.

        Params:
            cdf_vals: Discretized y values for the CDF
            utility_func: The utility function U(y, d)
            decision: The decision to consider

        Returns:
            expected utility: Approximate expected utility for the decision.
        """
        # Compute the utilities
        utilities = np.array([utility_func(y, decision) for y in cdf_vals])

        # Compute CDF jumps
        cdf_len = len(cdf_vals)
        delta_Q_star = np.diff(np.array([i / cdf_len for i in range(cdf_len)]))

        # Compute expected utility
        expected_utility = np.sum(utilities[:-1] * delta_Q_star)

        return expected_utility

    @staticmethod
    def optimal_decision_making(Decisions, y_test, utility_func):
        optimal_decisions, utilities = zip(
            *[
                max(((d, utility_func(y, d)) for d in Decisions), key=lambda x: x[1])
                for y in y_test
            ]
        )
        
        return optimal_decisions, utilities
    
    @staticmethod
    def make_decisions(expected_utilities, utility_func, y_test):
        decisions_made = []
        utilities = []
        for i, expected_utility in enumerate(zip(*expected_utilities)):
            max_utility = max(expected_utility)
            decision = expected_utility.index(max_utility)
            decisions_made.append(decision)
            utilities.append(utility_func(y_test[i], decision))
            
        return decisions_made, utilities