import numpy as np
from sklearn.base import clone
from .Utility import Utility

class BDT:
    @staticmethod
    def inductive_v1(model, Decisions, utility_func, X_train, y_train, X_test, y_test):
        expected_utilities = []
        for d in Decisions:
            y_train_d = Utility.create_utility_sequence(y_train, d, utility_func)
            model.fit(X_train, y_train_d)
            expected_utilities.append(model.predict(X_test))
        
        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)
        
        return utilities
    
    def inductive_v2(model, Decisions, utility_func, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        expected_utilities = []
        for d in Decisions:
            expected_utilities_d = Utility.create_utility_sequence(preds, d, utility_func)
            expected_utilities.append(expected_utilities_d)
        
        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)
        
        return utilities

    @staticmethod
    def online_v1(model, Decisions, utility_func, X_train, y_train, X_test, y_test):
        expected_utilities = []
        for d in Decisions:
            y_train_d = Utility.create_utility_sequence(y_train, d, utility_func)
            y_test_d = Utility.create_utility_sequence(y_test, d, utility_func)

            X_seen = X_train
            y_seen = y_train_d

            expected_utilities_d = []
            for i in range(len(X_test)):
                # TODO see if we can find a package that supports online bayesian methods natively.
                model_i = clone(model)
                model_i.fit(X_seen, y_seen)
                expected_utility = model_i.predict(X_test[i].reshape(1, -1))
                expected_utilities_d.append(expected_utility)

                X_seen = np.vstack((X_seen, X_test[i]))
                y_seen = np.hstack((y_seen, y_test_d[i]))

            expected_utilities.append(expected_utilities_d)

        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)

        return utilities
  
    @staticmethod
    def online_v2(model, Decisions, utility_func, X_train, y_train, X_test, y_test):
        X_seen = X_train
        y_seen = y_train
        
        expected_utilities = []
        preds = []
        for i in range(len(X_test)):
            model_i = clone(model)
            model_i.fit(X_seen, y_seen)
            pred = model_i.predict(X_test[i].reshape(1, -1))

            preds.append(pred)

            X_seen = np.vstack((X_seen, X_test[i]))
            y_seen = np.hstack((y_seen, y_test[i]))
            
        for d in Decisions:
            expected_utility_d = Utility.create_utility_sequence(preds, d, utility_func)
            expected_utilities.append(expected_utility_d)

        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)
        
        return utilities