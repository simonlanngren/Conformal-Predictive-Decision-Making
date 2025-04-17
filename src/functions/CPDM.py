import numpy as np
from crepes import WrapRegressor
from copy import deepcopy
from online_cp.CPS import NearestNeighboursPredictionMachine, KernelRidgePredictionMachine
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import root_scalar
from .Utility import Utility
from .model_selection import ModelSelection

class CPDM:
    @staticmethod
    def inductive_v1(model, Decisions, utility_func, X_train, y_train, X_cal, y_cal, X_test, y_test):
        expected_utilities = []
        for d in Decisions:
            # Create the new sequences
            y_train_d = Utility.create_utility_sequence(y_train, d, utility_func)
            y_cal_d   = Utility.create_utility_sequence(y_cal, d, utility_func)

            # Train model
            model.fit(X_train, y_train_d)

            # Wrap the model with conformal prediction
            cps = WrapRegressor(model)
            cps.calibrate(X_cal, y_cal_d, cps=True)

            # Get CPDs for the test samples
            cpds = cps.predict_cps(X_test, return_cpds=True)
            
            # Compute the expected utility
            expected_utilities_d = []
            for i in range(len(X_test)):
                expected_utility = Utility.compute_expected_utility(cpds[i], utility_func, d)
                expected_utilities_d.append(expected_utility)

            expected_utilities.append(expected_utilities_d)
            
        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)

        return utilities
    
    @staticmethod
    def inductive_v2(model, Decisions, utility_func, X_train, y_train, X_cal, y_cal, X_test, y_test):
        # Train the model
        model.fit(X_train, y_train)

        # Wrap the model with conformal prediction
        cps = WrapRegressor(model)
        cps.calibrate(X_cal, y_cal, cps=True)

        # Get CPDs for the test samples
        cpds = cps.predict_cps(X_test, return_cpds=True)
        
        expected_utilities = []
        for d in Decisions:
            # Compute the expected utility
            expected_utilities_d = []
            for i in range(len(X_test)):
                expected_utility = Utility.compute_expected_utility(cpds[i], utility_func, d)
                expected_utilities_d.append(expected_utility)

            expected_utilities.append(expected_utilities_d)
                        
        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)

        return utilities

    @staticmethod
    def online_v1(cps, Decisions, utility_func, X_train, y_train, X_test, y_test, search_space, n_splits=5, random_state=None):
        expected_utilities = []
        for i, d in enumerate(Decisions):
            # Create the new sequences
            y_train_d = Utility.create_utility_sequence(y_train, d, utility_func).astype(float)
            y_test_d  = Utility.create_utility_sequence(y_test, d, utility_func).astype(float)
            X_seen = X_train
            y_seen = y_train_d

            if isinstance(cps, NearestNeighboursPredictionMachine):
                y_seen   += np.random.normal(scale=1e-6, size=y_seen.shape)
                y_test_d += np.random.normal(scale=1e-6, size=y_test_d.shape)
                
                best_k = ModelSelection.online_cpdm_model_selection_knn(
                    X_seen,
                    y_seen,
                    search_space=search_space,
                    n_splits=n_splits,
                    random_state=random_state
                )
                cps_d = NearestNeighboursPredictionMachine(k=best_k)
            elif isinstance(cps, KernelRidgePredictionMachine):
                best_params = ModelSelection.online_cpdm_model_selection_krr(
                    X_seen,
                    y_seen,
                    search_space=search_space,
                    n_splits=n_splits,
                    random_state=random_state
                )
                kernel = C(best_params['kernel__k1__constant_value']) * RBF(length_scale=best_params['kernel__k2__length_scale'])
                cps_d = KernelRidgePredictionMachine(kernel=kernel, a=best_params['alpha'])
            else:
                cps_d = deepcopy(cps)
            
            cps_d.learn_initial_training_set(X_seen, y_seen)

            expected_utilities_d = []
            for x, y in zip(X_test, y_test_d):
                # Produce conformal predictive distribution
                cpd, precomputed = cps_d.predict_cpd(x=x, return_update=True)

                # Compute the expected utility from the cpd
                expected_utility = Utility.compute_expected_utility(cpd.y_vals, utility_func, d)
                expected_utilities_d.append(expected_utility)

                # Learn new object
                cps_d.learn_one(x=x, y=y, precomputed=precomputed)  # We pass precomputed as an argument to avoid redundant computations
                
                X_seen = np.append(X_seen, [x], axis=0)
                y_seen = np.append(y_seen, [y])
                    
                if isinstance(cps, NearestNeighboursPredictionMachine):
                    best_k = ModelSelection.online_cpdm_model_selection_knn(
                        X_seen,
                        y_seen,
                        search_space=search_space,
                        n_splits=n_splits,
                        random_state=random_state
                    )
                    cps_d.k = best_k
                
                if isinstance(cps, KernelRidgePredictionMachine):                    
                    best_params = ModelSelection.online_cpdm_model_selection_krr(
                        X_seen,
                        y_seen,
                        search_space=search_space,
                        n_splits=n_splits,
                        random_state=random_state
                    )
                    cps_d.kernel = C(best_params['kernel__k1__constant_value']) * RBF(length_scale=best_params['kernel__k2__length_scale'])
                    cps_d.a = best_params['alpha']

            expected_utilities_d = []
            expected_utilities.append(expected_utilities_d)

        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)

        return utilities
    
    @staticmethod
    def online_v2(cps, Decisions, utility_func, X_train, y_train, X_test, y_test, search_space, n_splits=5, random_state=None):
        X_seen = X_train
        y_seen = y_train
        
        if isinstance(cps, NearestNeighboursPredictionMachine):
            # Hyperparameter tuning for k
            best_k = ModelSelection.online_cpdm_model_selection_knn(
                X_seen,
                y_seen,
                search_space=search_space,
                n_splits=n_splits,
                random_state=random_state
            )
            chosen_cps = NearestNeighboursPredictionMachine(k=best_k)
        elif isinstance(cps, KernelRidgePredictionMachine):
            # Hyperparameter tuning
            best_params = ModelSelection.online_cpdm_model_selection_krr(
                X_seen,
                y_seen,
                search_space=search_space,
                n_splits=n_splits,
                random_state=random_state
            )
            kernel = C(best_params['kernel__k1__constant_value']) * RBF(length_scale=best_params['kernel__k2__length_scale'])
            chosen_cps = KernelRidgePredictionMachine(kernel=kernel, a=best_params['alpha'])
        else:
            chosen_cps = deepcopy(cps)
            
        chosen_cps.learn_initial_training_set(X_seen, y_seen)
        
        expected_utilities = [[],[]]
        for x, y in zip(X_test, y_test):
            # Produce conformal predictive distribution
            cpd, precomputed = chosen_cps.predict_cpd(x=x, return_update=True)
            
            for i, d in enumerate(Decisions):
                # Compute the expected utility from the cpd
                expected_utility = Utility.compute_expected_utility(cpd.y_vals, utility_func, d)
                expected_utilities[i].append(expected_utility)
            
            # Learn new object
            chosen_cps.learn_one(x=x, y=y, precomputed=precomputed)
            
            X_seen = np.append(X_seen, [x], axis=0)
            y_seen = np.append(y_seen, [y])
            
            if isinstance(cps, NearestNeighboursPredictionMachine):
                best_k = ModelSelection.online_cpdm_model_selection_knn(
                    X_seen,
                    y_seen,
                    search_space=search_space,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                
                chosen_cps.k = best_k 
            
            if isinstance(cps, KernelRidgePredictionMachine):
                best_params = ModelSelection.online_cpdm_model_selection_krr(
                    X_seen,
                    y_seen,
                    search_space=search_space,
                    n_splits=n_splits,
                    random_state=random_state
                )
                chosen_cps.kernel = C(best_params['kernel__k1__constant_value']) * RBF(length_scale=best_params['kernel__k2__length_scale'])
                chosen_cps.a = best_params['alpha']
        
        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)

        return utilities

    @staticmethod
    def compute_significance_and_coverage(h):
        def equation(epsilon):
            return 2 * h * epsilon**2 + np.log(epsilon)
        
        result = root_scalar(equation, bracket=[1e-10, 1], method='brentq')
        
        return round(result.root, 2)