import numpy as np
from copy import deepcopy
from tqdm.notebook import tqdm
from .functions.Utility import Utility

class BDT:
    @staticmethod
    def inductive_BDT(Decisions, X_train, y_train, X_test, y_test, utility_func, model):
        expected_utilities = []
        for d in Decisions:
            y_train_d = Utility.create_utility_sequence(y_train, d, utility_func)
            model.fit(X_train, y_train_d)
            expected_utilities.append(model.predict(X_test))

        average_utility, decisions_made = Utility.compute_average_utility(expected_utilities, utility_func, y_test)

        return average_utility, decisions_made

    @staticmethod
    def online_BDT(Decisions, X_train, y_train, X_test, y_test, utility_func, model):
        expected_utilities = []
        for d in Decisions:
            y_train_d = Utility.create_utility_sequence(y_train, d, utility_func)
            y_test_d = Utility.create_utility_sequence(y_test, d, utility_func)

            X_seen = X_train
            y_seen = y_train_d

            expected_utilities_d = []
            for i in tqdm(range(len(X_test)), desc="Processing Samples"):
                # TODO see if we can find a package that supports online bayesian methods natively.
                model_i = deepcopy(model)
                model_i.fit(X_seen, y_seen)
                expected_utility = model_i.predict(X_test[i].reshape(1, -1))
                expected_utilities_d.append(expected_utility)

                X_seen = np.vstack((X_seen, X_test[i]))
                y_seen = np.hstack((y_seen, y_test_d[i]))

            expected_utilities.append(expected_utilities_d)

        average_utility, decisions_made = Utility.compute_average_utility(expected_utilities, utility_func, y_test)

        return average_utility, decisions_made
