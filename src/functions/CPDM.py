import numpy as np
from crepes import WrapRegressor
from tqdm.notebook import tqdm
from online_cp.martingale import PluginMartingale
from copy import deepcopy
from online_cp.CPS import NearestNeighboursPredictionMachine
from sklearn.model_selection import KFold


class CPDM:
    @staticmethod
    def calibrate_model(X_cal, y_cal, model):
        cps_std = WrapRegressor(model)
        cps_std.calibrate(X_cal, y_cal, cps=True)
        return cps_std

    @staticmethod
    def compute_percentiles(cpd):
        cpd_len = len(cpd)
        percentiles = np.array([i / cpd_len for i in range(cpd_len)])
        return percentiles

    @staticmethod
    def find_median(percentiles):
        median_idx = np.where(percentiles >= 0.50)[0][0]
        return median_idx

    @staticmethod
    def find_percentile_range(percentiles, alpha=0.05):
        lower_idx = np.where(percentiles <= alpha / 2.0)[0][-1]
        upper_idx = np.where(percentiles >= 1 - alpha / 2.0)[0][0]
        return lower_idx, upper_idx

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
    def inductive_CPDM(
        Decisions, X_train, y_train, X_cal, y_cal, X_test, y_test, utility_func, model
    ):
        expected_utilities = []
        for d in Decisions:
            # Create the new sequences
            y_train_d = CPDM.create_utility_sequence(y_train, d, utility_func)
            y_cal_d = CPDM.create_utility_sequence(y_cal, d, utility_func)

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
                expected_utility = CPDM.compute_expected_utility(
                    cpds[i], utility_func, d
                )
                expected_utilities_d.append(expected_utility)

            expected_utilities.append(expected_utilities_d)

        # Decision making
        decisions_made = []
        cumulative_sum = 0
        average_utility = []
        for i, expected_utility in enumerate(zip(*expected_utilities), start=1):
            # Make decision
            max_utility = max(expected_utility)
            decision = expected_utility.index(max_utility)
            decisions_made.append(decision)

            # Compute average utility
            cumulative_sum += utility_func(y_test[i - 1], decision)
            average_utility.append(cumulative_sum / i)

        return decisions_made, average_utility

    @staticmethod
    def online_CPDM(
        Decisions, X_train, y_train, X_test, y_test, utility_func, epsilon, cps
    ):
        res = np.zeros(shape=(len(Decisions), len(X_test), 8))

        expected_utilities = []

        for idx, d in enumerate(Decisions):
            # Create the new sequences
            y_train_d = CPDM.create_utility_sequence(y_train, d, utility_func).astype(
                float
            )
            y_test_d = CPDM.create_utility_sequence(y_test, d, utility_func).astype(
                float
            )
            
            X_seen = X_train
            y_seen = y_train_d

            # Add small noise if the type condition is met
            if isinstance(cps, NearestNeighboursPredictionMachine):
                y_train_d += np.random.normal(scale=1e-6, size=y_train_d.shape)
                y_test_d += np.random.normal(scale=1e-6, size=y_test_d.shape)

                # hyperparameter tuning for k
                #best_k = CPDM.KFold_knn(X_seen, y_seen, epsilon)
                best_k = 5
                cps_d = NearestNeighboursPredictionMachine(k=best_k)  # Use optimal k
                
            else:
                cps_d = deepcopy(cps)

            cps_d.learn_initial_training_set(X_seen, y_seen)

            martingale = PluginMartingale(warning_level=100)

            Err = 0

            expected_utilities_d = []

            for i, (object, label) in tqdm(
                enumerate(zip(X_test, y_test_d)),
                total=y_test_d.shape[0],
                desc="Running online conformal prediction",
            ):
                # Reality outputs object and a random number tau
                tau = np.random.uniform(0, 1)

                # Forecaster outputs a conformal predictive distribution
                cpd, precomputed = cps_d.predict_cpd(
                    x=object, return_update=True
                )  # We return the precomputed update for later use

                # Which can be used to compute a prediction set
                Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)

                # Observe error
                err = cpd.err(Gamma=Gamma, y=label)
                Err += err

                cpd_vals = [
                    val for val in cpd.y_vals if 2.5 <= val <= 9.5
                ]  # Hard coded for this utility function

                # And perhaps the median is interesting too
                median = cpd.quantile(0.5, tau=tau)

                # Compute the expected utility from the cpd
                expected_utility = CPDM.compute_expected_utility(
                    cpd_vals, utility_func, d
                )

                expected_utilities_d.append(expected_utility)

                # Learn new object
                cps_d.learn_one(
                    x=object, y=label, precomputed=precomputed
                )  # We pass precomputed as an argument to avoid redundant computations

                # Compute p-value
                p = cpd(y=label, tau=tau)

                # Update martingale
                martingale.update_martingale_value(p)

                res[idx, i, 0] = Gamma.lower
                res[idx, i, 1] = Gamma.upper
                res[idx, i, 2] = label
                res[idx, i, 3] = err
                res[idx, i, 4] = Err
                res[idx, i, 5] = cpd.width(Gamma)  # Simple efficiency criterion for interval predictions
                res[idx, i, 6] = martingale.logM
                res[idx, i, 7] = median
                
                # Update X_seen and y_seen to fo hyperparameter tuning for knn
                if isinstance(cps, NearestNeighboursPredictionMachine):
                    X_seen = np.append(X_seen, [object], axis=0)
                    y_seen = np.append(y_seen, [label])

                    # hyperparameter tuning for k
                    #best_k = CPDM.KFold_knn(X_seen, y_seen, epsilon)
                    best_k = 5
                    cps_d.k = best_k
                    
            expected_utilities.append(expected_utilities_d)

        # Decision making
        decisions_made = []
        cumulative_sum = 0
        average_utility = []
        for i, expected_utility in enumerate(zip(*expected_utilities), start=1):
            # Make decision
            max_utility = max(expected_utility)
            decision = expected_utility.index(max_utility)
            decisions_made.append(decision)

            # Compute average utility
            cumulative_sum += utility_func(y_test[i - 1], decision)
            average_utility.append(cumulative_sum / i)

        return decisions_made, average_utility, res

    @staticmethod
    def KFold_knn(X_train, y_train, epsilon, k_values=[100]):
        best_k, best_score = None, float("inf")

        kf = KFold(n_splits=5, shuffle=True, random_state=2025)

        for k in k_values:
            cv_errors = []

            for train_idx, val_idx in kf.split(X_train):
                X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
                y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

                # Train model with current k
                cps_cv = NearestNeighboursPredictionMachine(k=k)
                cps_cv.learn_initial_training_set(X_train_cv, y_train_cv)

                errors = []
                for x, y in zip(X_val_cv, y_val_cv):
                    tau = np.random.uniform(0, 1)
                    cpd = cps_cv.predict_cpd(x=x)
                    Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
                    errors.append(cpd.err(Gamma=Gamma, y=y))

                cv_errors.append(np.mean(errors))

            avg_error = np.mean(cv_errors)
            if avg_error < best_score:
                best_score = avg_error
                best_k = k
        return best_k
