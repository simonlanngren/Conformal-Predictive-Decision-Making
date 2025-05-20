"""
cpdm.py

Implements Conformal Predictive Decision Making (CPDM) for the inductive
and online setting. Supports adaptive model selection for the conformalized
versions of Nearest Neighbours and Kernel Ridge Regressors methods in the
online setting.

Authors: Simon Lanngren (simlann@chalmers.se), Martin Toremark (toremark@chalmers.se)
"""

import numpy as np
from crepes import WrapRegressor
from copy import deepcopy
from online_cp.CPS import (
    NearestNeighboursPredictionMachine,
    KernelRidgePredictionMachine,
)
from sklearn.gaussian_process.kernels import RBF
from .utility import Utility
from .model_selection import ModelSelection


class CPDM:
    """
    Provides static methods for utility-optimized decision-making using conformal
    predictive systems (CPSs) in inductive and online learning settings.
    """

    @staticmethod
    def inductive(
        model, Decisions, utility_func, X_train, y_train, X_cal, y_cal, X_test, y_test
    ):
        """
        Inductive CPDM using a conformal wrapper around a fitted model.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            Scikit-learn compatible model.
        Decisions : set
            A set of possible decision values to evaluate.
        utility_func : callable
            A function that computes the utility given a prediction and a decision.
        X_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray
            Ground truth labels for training.
        X_cal : np.ndarray
            Feature matrix for calibration.
        y_cal : np.ndarray
            Ground truth labels for calibration.
        X_test : np.ndarray
            Feature matrix for testing.
        y_test : np.ndarray
            Ground truth labels for testing.

        Returns
        -------
        list of float
            Utility values for each decision made.
        """
        model.fit(X_train, y_train)
        cps = WrapRegressor(model)
        cps.calibrate(X_cal, y_cal, cps=True)
        cpds = cps.predict_cps(X_test, return_cpds=True)

        expected_utilities = []
        for d in Decisions:
            expected_utilities_d = []
            for i in range(len(X_test)):
                expected_utility = Utility.compute_expected_utility(
                    cpds[i], utility_func, d
                )
                expected_utilities_d.append(expected_utility)

            expected_utilities.append(expected_utilities_d)

        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)

        return utilities

    @staticmethod
    def online_v1(
        cps,
        Decisions,
        utility_func,
        X_train,
        y_train,
        X_test,
        y_test,
        search_space,
        n_splits=5,
        random_state=None,
    ):
        """
        Earlier Propoesed Online Conformal Predictive Decision Making (CPDM) based on per-decision models.

        For each decision value, a separate conformal predictive system (CPS) is trained using a
        transformed utility sequence. The system is updated sequentially, either by incremental learning
        or by retraining with updated hyperparameters after each new observation. Expected utilities are
        computed from conformal predictive distributions and used to select decisions.

        This implementation follows the method introduced by Vovk & Bendtsen (2018).

        Reference
        ---------
        Vovk, V., & Bendtsen, C. (2018). Conformal predictive decision making. In *Proceedings of the
        Seventh Workshop on Conformal and Probabilistic Prediction and Applications* (pp. 52–62). PMLR.
        URL: https://proceedings.mlr.press/v91/vovk18b.html

        Parameters
        ----------
        cps : object
            Initial conformal predictive system.
        Decisions : set
            A set of possible decision values to evaluate.
        utility_func : callable
            A function that computes the utility given a prediction and a decision.
        X_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray
            Ground truth labels for training.
        X_test : np.ndarray
            Feature matrix for testing.
        y_test : np.ndarray
            Ground truth labels for testing.
        search_space : dict
            Hyperparameter search space for model selection.
        n_splits : int, optional
            Number of folds used in cross-validation.
        random_state : int or None, optional
            Seed for reproducibility.

        Returns
        -------
        list of float
            Utility values for each decision made.
        """
        expected_utilities = []
        for i, d in enumerate(Decisions):
            y_train_d = Utility.create_utility_sequence(
                y_train, d, utility_func
            ).astype(float)
            y_test_d = Utility.create_utility_sequence(y_test, d, utility_func).astype(
                float
            )
            X_seen = X_train
            y_seen = y_train_d

            if isinstance(cps, NearestNeighboursPredictionMachine):
                y_seen += np.random.normal(scale=1e-6, size=y_seen.shape)
                y_test_d += np.random.normal(scale=1e-6, size=y_test_d.shape)

                best_k = ModelSelection.online_cpdm_model_selection_knn(
                    X_seen,
                    y_seen,
                    search_space=search_space,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                cps_d = NearestNeighboursPredictionMachine(k=best_k)
            elif isinstance(cps, KernelRidgePredictionMachine):
                best_params = ModelSelection.online_cpdm_model_selection_krr(
                    X_seen,
                    y_seen,
                    search_space=search_space,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                kernel = RBF(length_scale=best_params["kernel__length_scale"])
                cps_d = KernelRidgePredictionMachine(
                    kernel=kernel, a=best_params["alpha"]
                )
            else:
                cps_d = deepcopy(cps)

            cps_d.learn_initial_training_set(X_seen, y_seen)

            expected_utilities_d = []
            for x, y in zip(X_test, y_test_d):
                cpd, precomputed = cps_d.predict_cpd(x=x, return_update=True)

                expected_utility = Utility.compute_expected_utility(
                    cpd.y_vals, utility_func, d
                )
                expected_utilities_d.append(expected_utility)

                X_seen = np.append(X_seen, [x], axis=0)
                y_seen = np.append(y_seen, [y])

                if isinstance(cps, KernelRidgePredictionMachine):
                    best_params = ModelSelection.online_cpdm_model_selection_krr(
                        X_seen,
                        y_seen,
                        search_space=search_space,
                        n_splits=n_splits,
                        random_state=random_state,
                    )
                    kernel = RBF(length_scale=best_params["kernel__length_scale"])
                    cps_d = KernelRidgePredictionMachine(
                        kernel=kernel, a=best_params["alpha"]
                    )
                    cps_d.learn_initial_training_set(X_seen, y_seen)
                else:
                    cps_d.learn_one(x=x, y=y, precomputed=precomputed)

                if isinstance(cps, NearestNeighboursPredictionMachine):
                    best_k = ModelSelection.online_cpdm_model_selection_knn(
                        X_seen,
                        y_seen,
                        search_space=search_space,
                        n_splits=n_splits,
                        random_state=random_state,
                    )
                    cps_d.k = best_k

            expected_utilities.append(expected_utilities_d)

        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)

        return utilities

    @staticmethod
    def online_v2(
        cps,
        Decisions,
        utility_func,
        X_train,
        y_train,
        X_test,
        y_test,
        search_space,
        n_splits=5,
        random_state=None,
    ):
        """
        Online Conformal Predictive Decision Making (CPDM) with a shared predictive system.

        Uses a single conformal predictive system (CPS) across all decision values.
        The CPS is updated sequentially after each test observation using either incremental
        updates or full retraining with updated hyperparameters. For every new observation,
        expected utilities are computed across all decision values and used to determine
        the final utility outcomes.

        This version is more computationally efficient than `online_v1`, since it avoids
        duplicating CPSs per decision.

        Parameters
        ----------
        cps : object
            Initial conformal predictive system.
        Decisions : set
            A set of possible decision values to evaluate.
        utility_func : callable
            A function that computes the utility given a prediction and a decision.
        X_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray
            Ground truth labels for training.
        X_test : np.ndarray
            Feature matrix for testing.
        y_test : np.ndarray
            Ground truth labels for testing.
        search_space : dict
            Hyperparameter search space for model selection.
        n_splits : int, optional
            Number of folds used in cross-validation.
        random_state : int or None, optional
            Seed for reproducibility.

        Returns
        -------
        list of float
            Utility values for each decision made.
        """
        X_seen = X_train
        y_seen = y_train

        if isinstance(cps, NearestNeighboursPredictionMachine):
            best_k = ModelSelection.online_cpdm_model_selection_knn(
                X_seen,
                y_seen,
                search_space=search_space,
                n_splits=n_splits,
                random_state=random_state,
            )
            chosen_cps = NearestNeighboursPredictionMachine(k=best_k)
        elif isinstance(cps, KernelRidgePredictionMachine):
            best_params = ModelSelection.online_cpdm_model_selection_krr(
                X_seen,
                y_seen,
                search_space=search_space,
                n_splits=n_splits,
                random_state=random_state,
            )
            kernel = RBF(length_scale=best_params["kernel__length_scale"])
            chosen_cps = KernelRidgePredictionMachine(
                kernel=kernel, a=best_params["alpha"]
            )
        else:
            chosen_cps = deepcopy(cps)

        chosen_cps.learn_initial_training_set(X_seen, y_seen)

        expected_utilities = [[] for _ in Decisions]
        for i, (x, y) in enumerate(zip(X_test, y_test)):
            cpd, precomputed = chosen_cps.predict_cpd(x=x, return_update=True)

            for i, d in enumerate(Decisions):
                expected_utility = Utility.compute_expected_utility(
                    cpd.y_vals, utility_func, d
                )
                expected_utilities[i].append(expected_utility)

            X_seen = np.append(X_seen, [x], axis=0)
            y_seen = np.append(y_seen, [y])

            if isinstance(chosen_cps, KernelRidgePredictionMachine):
                best_params = ModelSelection.online_cpdm_model_selection_krr(
                    X_seen,
                    y_seen,
                    search_space=search_space,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                kernel = RBF(length_scale=best_params["kernel__length_scale"])
                chosen_cps = KernelRidgePredictionMachine(
                    kernel=kernel, a=best_params["alpha"]
                )
                chosen_cps.learn_initial_training_set(X_seen, y_seen)
            else:
                chosen_cps.learn_one(x=x, y=y, precomputed=precomputed)

            if isinstance(chosen_cps, NearestNeighboursPredictionMachine):
                best_k = ModelSelection.online_cpdm_model_selection_knn(
                    X_seen,
                    y_seen,
                    search_space=search_space,
                    n_splits=n_splits,
                    random_state=random_state,
                )

                chosen_cps.k = best_k

        _, utilities = Utility.make_decisions(expected_utilities, utility_func, y_test)

        return utilities
