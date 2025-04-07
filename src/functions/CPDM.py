import numpy as np
from crepes import WrapRegressor
from tqdm.notebook import tqdm
from online_cp.martingale import PluginMartingale
from copy import deepcopy
from online_cp.CPS import NearestNeighboursPredictionMachine
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from .functions.Utility import Utility

class CPDM:
    @staticmethod
    def inductive_CPDM(Decisions, X_train, y_train, X_cal, y_cal, X_test, y_test, utility_func, model):
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

            # Get CPDs for test samples
            cpds = cps.predict_cps(X_test, return_cpds=True)
            
            # Compute expected utility
            expected_utilities_d = []
            for i in range(len(X_test)):
                expected_utility = Utility.compute_expected_utility(cpds[i], utility_func, d)
                expected_utilities_d.append(expected_utility)

            expected_utilities.append(expected_utilities_d)
            
        average_utility, decisions_made = Utility.compute_average_utility(expected_utilities, utility_func, y_test)

        return average_utility, decisions_made

    @staticmethod
    def online_CPDM(Decisions, X_train, y_train, X_test, y_test, utility_func, cps, epsilon=0.05):
        n_plots = 8
        res = np.zeros(shape=(len(Decisions), len(X_test), n_plots))
        expected_utilities = []
        for i, d in enumerate(Decisions):
            # Create the new sequences
            y_train_d = Utility.create_utility_sequence(y_train, d, utility_func).astype(float)
            y_test_d  = Utility.create_utility_sequence(y_test, d, utility_func).astype(float)
            X_seen = X_train
            y_seen = y_train_d

            if isinstance(cps, NearestNeighboursPredictionMachine):
                # Ensure the labels are distinct for KNN
                y_seen   += np.random.normal(scale=1e-6, size=y_seen.shape)
                y_test_d += np.random.normal(scale=1e-6, size=y_test_d.shape)

                # Hyperparameter tuning for k
                best_k = CPDM.KFold_knn(X_seen, y_seen, epsilon)
                cps_d  = NearestNeighboursPredictionMachine(k=best_k)
            else:
                cps_d = deepcopy(cps)
            
            cps_d.learn_initial_training_set(X_seen, y_seen)

            martingale = PluginMartingale(warning_level=100)
            Err = 0
            expected_utilities_d = []
            for j, (x, y) in tqdm(enumerate(zip(X_test, y_test_d)), total=y_test_d.shape[0], desc="Running online conformal prediction"):
                # Generate a random number
                tau = np.random.uniform(0, 1)

                # Produce conformal predictive distribution
                cpd, precomputed = cps_d.predict_cpd(x=object, return_update=True)

                # Compute a prediction set
                Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)

                # Observe error
                err = cpd.err(Gamma=Gamma, y=y)
                Err += err

                # TODO Verify
                cpd_vals = [val for val in cpd.y_vals if 2.5 <= val <= 9.5] # Hard coded for this utility function

                # And perhaps the median is interesting too
                median = cpd.quantile(0.5, tau=tau)

                # Compute the expected utility from the cpd
                expected_utility = CPDM.compute_expected_utility(cpd_vals, utility_func, d)
                expected_utilities_d.append(expected_utility)

                # Learn new object
                cps_d.learn_one(x=x, y=y, precomputed=precomputed)  # We pass precomputed as an argument to avoid redundant computations

                # Compute p-value
                p = cpd(y=y, tau=tau)

                # Update martingale
                martingale.update_martingale_value(p)

                # Populate res
                res[i, j, 0] = Gamma.lower
                res[i, j, 1] = Gamma.upper
                res[i, j, 2] = y
                res[i, j, 3] = err
                res[i, j, 4] = Err
                res[i, j, 5] = cpd.width(Gamma)  # Simple efficiency criterion for interval predictions
                res[i, j, 6] = martingale.logM
                res[i, j, 7] = median
                
                # Update X_seen and y_seen for hyperparameter tuning of KNN
                if isinstance(cps, NearestNeighboursPredictionMachine):
                    X_seen = np.append(X_seen, [x], axis=0)
                    y_seen = np.append(y_seen, [y])
                    
                    best_k = CPDM.KFold_knn(X_seen, y_seen)
                    cps_d.k = best_k
            
            expected_utilities.append(expected_utilities_d)

        average_utility, decisions_made = Utility.compute_average_utility(expected_utilities, utility_func, y_test)

        return average_utility, decisions_made, res

    @staticmethod
    def KFold_knn(X_train, y_train, n_splits=5, random_state=2025, epsilon=0.05, k_values=[3, 5, 7, 10, 15, 40], n_jobs=-1):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(X_train))

        def eval_k(k):
            cv_errors = []
            for train_idx, val_idx in splits:
                X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
                y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

                cps_cv = NearestNeighboursPredictionMachine(k=k)
                cps_cv.learn_initial_training_set(X_train_cv, y_train_cv)

                errors = []
                for x, y in zip(X_val_cv, y_val_cv):
                    tau = np.random.uniform(0, 1)
                    cpd = cps_cv.predict_cpd(x=x)
                    Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
                    err = cpd.err(Gamma=Gamma, y=y)
                    errors.append(err)

                cv_errors.append(np.mean(errors))

            return k, np.mean(cv_errors)

        results = Parallel(n_jobs=n_jobs)(delayed(eval_k)(k) for k in k_values)
        best_k, _ = min(results, key=lambda x: x[1])
        return best_k
