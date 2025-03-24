import numpy as np
from copy import deepcopy
from tqdm.notebook import tqdm


class BDT:
    @staticmethod
    def create_utility_sequence(ys, decision, utility):
        return np.array([utility(y, decision) for y in ys])

    @staticmethod
    def compute_expected_utility(cpd_vals, utility_func, decision):
        """
        Computes the expected utility using the CPD function.

        Params:
            cpd_vals: Discretized y values for the CPD
            utility_func: The utility function U(y, d)
            decision: The decision to consider

        Returns:
            expected utility: Approximate expected utility for the decision.
        """
        # Compute the utilities
        utilities = np.array([utility_func(y, decision) for y in cpd_vals])

        # Compute CPD jumps
        cpd_len = len(cpd_vals)
        delta_Q_star = np.diff(np.array([i / cpd_len for i in range(cpd_len)]))

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
    def inductive_BDT(Decisions, X_train, y_train, X_test, y_test, utility_func, model):
        expected_utilities = []
        for d in Decisions:
            y_train_d = BDT.create_utility_sequence(y_train, d, utility_func)
            model.fit(X_train, y_train_d)
            expected_utilities.append(model.predict(X_test))

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

        return decisions_made, average_utility

    @staticmethod
    def online_BDT(Decisions, X_train, y_train, X_test, y_test, utility_func, model):
        expected_utilities = []
        for d in Decisions:
            y_train_d = BDT.create_utility_sequence(y_train, d, utility_func)
            y_test_d = BDT.create_utility_sequence(y_test, d, utility_func)

            X_seen = X_train
            y_seen = y_train_d

            expected_utilities_d = []

            for i in tqdm(range(len(X_test)), desc="Processing Samples"):
                model_i = deepcopy(model)

                model_i.fit(X_seen, y_seen)
                expected_utility = model_i.predict(X_test[i].reshape(1, -1))

                expected_utilities_d.append(expected_utility)

                X_seen = np.vstack((X_seen, X_test[i]))
                y_seen = np.hstack((y_seen, y_test_d[i]))

            expected_utilities.append(expected_utilities_d)

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

        return decisions_made, average_utility
