import numpy as np


class Utility:
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

        # Compute CPD jumps
        cdf_len = len(cdf_vals)
        delta_Q_star = np.diff(np.array([i / cdf_len for i in range(cdf_len)]))

        # Compute expected utility
        expected_utility = np.sum(utilities[:-1] * delta_Q_star)

        return expected_utility

    @staticmethod
    def optimal_decision_making(Decisions, y_test, utility_func):
        optimal_decisions, max_utilities = zip(
            *[
                max(((d, utility_func(y, d)) for d in Decisions), key=lambda x: x[1])
                for y in y_test
            ]
        )
        
        optimal_average_utility = [
            sum(max_utilities[: i + 1]) / (i + 1) for i in range(len(max_utilities))
        ]
        
        return optimal_decisions, optimal_average_utility
    
    @staticmethod
    def compute_average_utility(expected_utilities, utility_func, y_test):
        decisions_made = []
        cumulative_sum = 0
        average_utility = []
        for i, expected_utility in enumerate(zip(*expected_utilities), start=1):
            # Make decision
            max_utility = max(expected_utility)
            decision = expected_utility.index(max_utility)
            decisions_made.append(decision)

            # Compute cumulative average utility
            cumulative_sum += utility_func(y_test[i - 1], decision)
            average_utility.append(cumulative_sum / i)
            
        return average_utility, decisions_made