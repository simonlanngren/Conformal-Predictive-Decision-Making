import numpy as np
from copy import deepcopy


class PredictiveDecisionMaking:
    @staticmethod
    def inductive_predictive_decision_making(
        Decisions, X_train, y_train, X_test, y_test, utility_func, model, threshold
    ):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        decisions_made = np.where(y_pred >= threshold, 1, 0)
        utilities = [utility_func(y, d) for y, d in zip(y_test, decisions_made)]
        average_utility = np.cumsum(utilities) / np.arange(1, len(utilities) + 1)
        return decisions_made, average_utility

    @staticmethod
    def online_predictive_decision_making(
        Decisions, X_train, y_train, X_test, y_test, utility_func, model, threshold
    ):
        X_seen = X_train
        y_seen = y_train
        decisions_made = []
        for i, x in enumerate(X_test):
            model_i = deepcopy(model)
            model_i.fit(X_seen, y_seen)
            y_pred = model_i.predict(x.reshape(1, -1))
            decisions_made.append(np.where(y_pred >= threshold, 1, 0))
            X_seen = np.vstack((X_seen, x))
            y_seen = np.hstack((y_seen, y_test[i]))

        utilities = [utility_func(y, d) for y, d in zip(y_test, decisions_made)]
        average_utility = np.cumsum(utilities) / np.arange(1, len(utilities) + 1)
        return decisions_made, average_utility
