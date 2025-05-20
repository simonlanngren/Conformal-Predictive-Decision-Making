"""
model_selection.py

Provides methods for model selection, including general-purpose grid search
for the inductive setting and specialized procedures for conformalized KNN
and KRR in the online CPDM setting.

Authors: Simon Lanngren (simlann@chalmers.se), Martin Toremark (toremark@chalmers.se)
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from online_cp.CPS import (
    NearestNeighboursPredictionMachine,
    KernelRidgePredictionMachine,
)
from sklearn.gaussian_process.kernels import RBF
from joblib import Parallel, delayed
from itertools import product


class ModelSelection:
    """
    Static methods for hyperparameter tuning in inductive and online CPDM settings.
    """

    @staticmethod
    def model_selection(
        X_train,
        y_train,
        estimator,
        param_grid,
        n_splits=5,
        random_state=None,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    ):
        """
        Perform standard grid search cross-validation to find the best hyperparameters.

        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray
            Ground truth labels for training.
        estimator : sklearn.base.BaseEstimator
            Model to tune.
        param_grid : dict
            Grid of hyperparameters to search.
        n_splits : int, optional
            Number of folds used in cross-validation.
        random_state : int or None, optional
            Seed for reproducibility.
        n_jobs : int, optional
            Number of parallel jobs (default is -1, i.e., use all processors).
        scoring : str, optional
            Scoring metric for evaluation (default is 'neg_mean_squared_error').

        Returns
        -------
        tuple
            Best parameter dictionary and corresponding best score.
        """

        if "n_neighbors" in param_grid:
            param_grid["n_neighbors"] = [
                k
                for k in param_grid["n_neighbors"]
                if k <= len(X_train) * (1 - 1 / n_splits)
            ]

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            n_jobs=n_jobs,
            cv=cv,
            verbose=0,
            scoring=scoring,
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        return best_params, best_score

    @staticmethod
    def online_cpdm_model_selection_knn(
        X_train, y_train, search_space, n_splits=5, random_state=None
    ):
        """
        Custom hyperparameter tuning for conformalized KNN in online CPDM.

        Generates prediction sets from the CPD and evaluates their errors across
        cross-validation folds. The average error is used to select the optimal
        number of neighbors `k`.

        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray
            Ground truth labels for training.
        search_space : dict
            Dictionary with list of possible `n_neighbors` values.
        n_splits : int, optional
            Number of cross-validation splits.
        random_state : int or None, optional
            Seed for reproducibility.

        Returns
        -------
        int
            Best value for number of neighbors (k).
        """

        # Evaluate the error for a single (x, y) sample
        # using a randomly sampled tau and the conformal prediction set
        def eval_sample(cps_model, x, y, epsilon=0.05):
            tau = np.random.uniform(0, 1)
            cpd = cps_model.predict_cpd(x=x)
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            return cpd.err(Gamma=Gamma, y=y)

        # Evaluate a given value of k using cross-validation
        def eval_k(k, kf, X, y):
            cv_errors = []
            for train_idx, val_idx in kf.split(X):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                cps = NearestNeighboursPredictionMachine(k=k)
                cps.learn_initial_training_set(X_train, y_train)

                val_errors = Parallel(n_jobs=-1)(
                    delayed(eval_sample)(cps, x, y) for x, y in zip(X_val, y_val)
                )

                cv_errors.append(np.mean(val_errors))

            return k, np.mean(cv_errors)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        valid_ks = [
            k
            for k in search_space["n_neighbors"]
            if k <= len(X_train) * (1 - 1 / n_splits)
        ]

        results = Parallel(n_jobs=-1)(
            delayed(eval_k)(k, kf, X_train, y_train) for k in valid_ks
        )

        best_k, _ = min(results, key=lambda x: x[1])

        return best_k

    @staticmethod
    def online_cpdm_model_selection_krr(
        X_train, y_train, search_space, n_splits=5, random_state=None
    ):
        """
        Custom hyperparameter tuning for conformalized KRR using the RBF kernel
        in online CPDM.

        Generates prediction sets from the CPD and evaluates their errors across
        cross-validation folds. The average error is used to select the optimal
        combination of `alpha` and `kernel__length_scale`.

        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray
            Ground truth labels for training.
        search_space : dict
            Dictionary with lists of values for 'alpha' and 'kernel__length_scale'.
        n_splits : int, optional
            Number of cross-validation splits.
        random_state : int or None, optional
            Seed for reproducibility.

        Returns
        -------
        dict
            Dictionary containing the best 'alpha' and 'kernel__length_scale'.
        """

        # Evaluate the error for a single (x, y) sample
        # using a randomly sampled tau and the conformal prediction set
        def eval_sample(cps_model, x, y, epsilon=0.05):
            tau = np.random.uniform(0, 1)
            cpd = cps_model.predict_cpd(x=x)
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            return cpd.err(Gamma=Gamma, y=y)

        # Evaluate a specific combination of alpha and length_scale
        # using cross-validation
        def eval_params(params, kf, X, y):
            cv_errors = []
            kernel = RBF(length_scale=params["kernel__length_scale"])

            for train_idx, val_idx in kf.split(X):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                cps = KernelRidgePredictionMachine(kernel=kernel, a=params["alpha"])
                cps.learn_initial_training_set(X_tr, y_tr)

                val_errors = Parallel(n_jobs=-1)(
                    delayed(eval_sample)(cps, x, y) for x, y in zip(X_val, y_val)
                )

                cv_errors.append(np.mean(val_errors))

            return params, np.mean(cv_errors)

        # Setup
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        all_param_combinations = list(
            product(search_space["alpha"], search_space["kernel__length_scale"])
        )

        param_dicts = [
            {"alpha": alpha, "kernel__length_scale": length_scale}
            for alpha, length_scale in all_param_combinations
        ]

        results = Parallel(n_jobs=-1)(
            delayed(eval_params)(params, kf, X_train, y_train) for params in param_dicts
        )

        best_params, _ = min(results, key=lambda x: x[1])

        return best_params
