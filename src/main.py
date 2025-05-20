"""
main.py

Core experiment engine for running predictive decision-making simulations
under both inductive and online learning modes. Supports Conformal Predictive
Decision Making (CPDM), Bayesian Decision Theory (BDT), and Point Predictive 
Decision Making (PPDM).

Provides functionality for data loading, model training with optional
hyperparameter tuning, evaluation based on utility functions, and optional
saving and visualization of results.

Authors: Simon Lanngren (simlann@chalmers.se), Martin Toremark (toremark@chalmers.se)
"""

# Internal imports
from .functions.utility import Utility
from .functions.cpdm import CPDM
from .functions.bdt import BDT
from .functions.predictive_decision_making import PredictiveBinaryDecisionMaking
from .functions.model_selection import ModelSelection
from .functions.data_generation import DataGeneration
from .functions.plots import Plots

# External imports
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import fmin_l_bfgs_b
from online_cp.CPS import (
    RidgePredictionMachine,
    NearestNeighboursPredictionMachine,
    KernelRidgePredictionMachine,
)
from tqdm import tqdm
import json
import os


class Main:
    """
    Main interface for running predictive decision-making experiments.

    Handles the full pipeline: data splitting, model training, utility evaluation,
    plotting, and result export. Supports CPDM v1/v2, PPDM, and BDT methods
    across inductive and online settings.

    Parameters
    ----------
    threshold : float
        Classification threshold used in predictive decision making.
    tp : float
        Utility value for a true positive.
    tn : float
        Utility value for a true negative.
    fp : float
        Utility value for a false positive.
    fn : float
        Utility value for a false negative.
    experiment_name : str
        Unique identifier used when saving plots and experiment outputs.
    """
    
    def __init__(self, threshold, tp, tn, fp, fn, experiment_name):
        self.Decisions = {0, 1}
        self.utility_func = Utility.create_utility_func(threshold, tp, tn, fp, fn)
        self.threshold = threshold
        self.experiment_name = experiment_name

    def run_experiment(
        self,
        data,
        # experiment_config
        output_folder="data",
        save_experiment=True,
        # method_config
        mode="Inductive",
        run_v1=False,
        run_v2=True,
        run_predictive=False,
        # data_config
        n_runs=1000,
        sample_size=1000,
        test_size=0.2,
        cal_size=0.2,
        random_state=None,
        # models_config
        include_ridge=True,
        include_knn=True,
        include_krr=True,
        include_bayesian_ridge=True,
        include_gp=True,
        # plot_config
        plot_distributions=True,
        print_split=True,
        plot_average_utility=True,
        plot_cumulative_regret=True,
        plot_confidence=0.95,
        store_target_plot=False,
        # model_selection_config
        n_splits=5,
        search_space_knn=None,
        search_space_ridge=None,
        search_space_krr=None,
    ):
        """
        Run a full experiment using the specified configuration and methods.

        This method performs the following steps:
        - Optionally plots the target and feature distributions
        - Executes a predefined number of bootstrap runs of the experiment
        - Splits the data into prope training, train, calibration, and test sets
        - Runs inductive or online decision-making methods
        - Aggregates results and optionally plots and saves them

        Parameters
        ----------
        data : pandas.DataFrame
            The full dataset to be used in the experiment. Must include a 'Target' column.
        output_folder : str, optional
            Folder where plots and results should be stored (default is 'data').
        save_experiment : bool, optional
            Whether to save the result JSON file (default is True).
        mode : {'Inductive', 'Online'}, optional
            Mode of the experiment (default is 'Inductive').
        run_v1 : bool, optional
            Whether to run CPDM v1 in online setting (default is False).
        run_v2 : bool, optional
            Whether to run CPDM v2 in online setting (default is True).
        run_predictive : bool, optional
            Whether to run Point Predictive Decision Making (PPDM) (default is False).
        n_runs : int, optional
            Number of bootstrap runs (default is 1000).
        sample_size : int, optional
            Size of each bootstrap sample (default is 1000).
        test_size : float, optional
            Proportion of data used for testing (default is 0.2).
        cal_size : float, optional
            Proportion of proper training set used for calibration (default is 0.2).
        random_state : int or None, optional
            Seed for reproducibility.
        include_ridge, include_knn, include_krr : bool, optional
            Whether to include Ridge, KNN, and KRR models.
        include_bayesian_ridge, include_gp : bool, optional
            Whether to include Bayesian Ridge and Gaussian Process models.
        plot_distributions : bool, optional
            Whether to plot target and feature distributions (default is True).
        print_split : bool, optional
            Whether to print the split sizes on the first run (default is True).
        plot_average_utility : bool, optional
            Whether to plot average utility (default is True).
        plot_cumulative_regret : bool, optional
            Whether to plot cumulative regret (default is True).
        plot_confidence : float, optional
            Confidence level for error bars in plots (default is 0.95).
        store_target_plot : bool, optional
            Whether to save the target distribution plot (default is False).
        n_splits : int, optional
            Number of folds for cross-validation (default is 5).
        search_space_knn, search_space_ridge, search_space_krr : dict or None, optional
            Hyperparameter search spaces for model selection.

        Returns
        -------
        None
        """
        if plot_distributions:
            plots_dir = os.path.join(output_folder, "plots")
            output_filename = f"{self.experiment_name}_target_distribution.pdf"
            Plots.plot_target_distribution(
                data,
                store_target_plot=store_target_plot,
                output_folder=plots_dir,
                output_filename=output_filename,
            )
            Plots.plot_first_feature_distribution(data)

        experiment = defaultdict(list)
        for i in tqdm(range(n_runs), desc="Processing runs"):
            bootstrap_data = data.sample(
                n=sample_size, replace=True, random_state=random_state
            )
            distinct_data = DataGeneration.add_epsilon(bootstrap_data)

            X = distinct_data.drop(columns=["Target"])
            y = distinct_data["Target"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size
            )
            X_proper, X_cal, y_proper, y_cal = train_test_split(
                X_train, y_train, test_size=cal_size
            )

            scaler = StandardScaler()
            if mode == "Inductive":
                X_proper = scaler.fit_transform(X_proper)
                X_cal = scaler.transform(X_cal)
            else:
                X_train = scaler.fit_transform(X_train)

            X_test = scaler.transform(X_test)

            def to_numpy_safe(x):
                return x.to_numpy() if hasattr(x, "to_numpy") else x

            X_train = to_numpy_safe(X_train)
            y_train = to_numpy_safe(y_train)
            X_proper = to_numpy_safe(X_proper)
            y_proper = to_numpy_safe(y_proper)
            X_cal = to_numpy_safe(X_cal)
            y_cal = to_numpy_safe(y_cal)
            X_test = to_numpy_safe(X_test)
            y_test = to_numpy_safe(y_test)

            if i == 0 and print_split:
                if mode == "Inductive":
                    print(f"Proper train set size: {len(y_proper)}")
                    print(f"Calibration set size: {len(y_cal)}")
                else:
                    print(f"Train set size: {len(y_train)}")
                print(f"Test set size: {len(y_test)}")
                print("----------------------------------------")

            if mode == "Inductive":
                res = self._inductive_setting(
                    X_proper,
                    y_proper,
                    X_train,
                    y_train,
                    X_cal,
                    y_cal,
                    X_test,
                    y_test,
                    run_predictive=run_predictive,
                    include_knn=include_knn,
                    include_ridge=include_ridge,
                    include_krr=include_krr,
                    n_splits=n_splits,
                    search_space_knn=search_space_knn,
                    search_space_ridge=search_space_ridge,
                    search_space_krr=search_space_krr,
                    include_bayesian_ridge=include_bayesian_ridge,
                    include_gp=include_gp,
                    random_state=random_state,
                )
            elif mode == "Online":
                res = self._online_setting(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    run_v1=run_v1,
                    run_v2=run_v2,
                    run_predictive=run_predictive,
                    include_knn=include_knn,
                    include_ridge=include_ridge,
                    include_krr=include_krr,
                    n_splits=n_splits,
                    search_space_knn=search_space_knn,
                    search_space_ridge=search_space_ridge,
                    search_space_krr=search_space_krr,
                    include_bayesian_ridge=include_bayesian_ridge,
                    include_gp=include_gp,
                    random_state=random_state,
                )
            else:
                raise ValueError(f"Unsupported learning mode: {mode}")

            for key, value in res.items():
                experiment[key].append(value)

            _, utilities = Utility.optimal_decision_making(
                self.Decisions, y_test, self.utility_func
            )
            experiment["Optimal"].append(utilities)

        plots_dir = os.path.join(output_folder, "plots")

        if plot_average_utility:
            output_filename = f"{self.experiment_name}_average_utility.pdf"
            Plots.plot_average_utility(
                experiment, plots_dir, output_filename, confidence=plot_confidence
            )

        if plot_cumulative_regret:
            output_filename = f"{self.experiment_name}_regret.pdf"
            Plots.plot_cumulative_regret(
                experiment, plots_dir, output_filename, confidence=plot_confidence
            )

        if save_experiment:
            raw_data_dir = os.path.join(output_folder, "raw_data")

            output_filename = f"{self.experiment_name}.json"
            json_path = os.path.join(raw_data_dir, output_filename)
            with open(json_path, "w") as f:
                json.dump(experiment, f, indent=4)
            print(f"Saved JSON → {json_path}")

    def _inductive_setting(
        self,
        X_proper,
        y_proper,
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        run_predictive=False,
        include_knn=True,
        include_ridge=True,
        include_krr=True,
        n_splits=5,
        search_space_knn=None,
        search_space_ridge=None,
        search_space_krr=None,
        include_bayesian_ridge=True,
        include_gp=True,
        random_state=None,
    ):
        """
        Run the inductive learning pipeline using CPDM, PPDM, and BDT models.

        Parameters
        ----------
        X_proper : np.ndarray
            Feature matrix for proper training set.
        y_proper : np.ndarray
            Ground truth labels for proper training set.
        X_train : np.ndarray
            Feature matrix for training dataset used for BDT and PPDM.
        y_train : np.ndarray
            Ground truth labels for training set.
        X_cal : np.ndarray
            Feature matrix for calibration set.
        y_cal : np.ndarray
            Ground truth labels for calibration.
        X_test : np.ndarray
            Feature matrix for proper testing.
        y_test : np.ndarray
            Ground truth labels for testing.
        run_v1, run_v2, run_predictive : bool
            Flags for enabling different methods.
        include_knn, include_ridge, include_krr : bool
            Flags for enabling frequentist models used for CPDM and PPDM.
        include_bayesian_ridge, include_gp : bool
            Flags for enabling Bayesian models used for BDT.
        n_splits : int
            Number of folds for model selection.
        search_space_knn, search_space_ridge, search_space_krr : dict or None
            Hyperparameter grids for model selection.
        random_state : int or None
            Seed for reproducibility.

        Returns
        -------
        dict
            Dictionary mapping method names to their utility values.
        """
        models = []

        models = self._select_frequentist_models(
            X_train=X_proper,
            y_train=y_proper,
            include_knn=include_knn,
            include_ridge=include_ridge,
            include_krr=include_krr,
            n_splits=n_splits,
            search_space_knn=search_space_knn,
            search_space_ridge=search_space_ridge,
            search_space_krr=search_space_krr,
            random_state=random_state,
        )

        bayesian_models = []
        if include_bayesian_ridge:
            bayesian_models.append(BayesianRidge())

        if include_gp:

            def custom_optimizer(obj_func, initial_theta, bounds):
                theta_opt, func_min, _ = fmin_l_bfgs_b(
                    obj_func, initial_theta, bounds=bounds, maxiter=50000
                )
                return theta_opt, func_min

            gp = GaussianProcessRegressor(
                kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-10, 1e10)),
                optimizer=custom_optimizer,
                normalize_y=True,
                random_state=random_state,
            )
            bayesian_models.append(gp)

        res = {}
        for model in models:
            utilities = CPDM.inductive(
                model,
                self.Decisions,
                self.utility_func,
                X_proper,
                y_proper,
                X_cal,
                y_cal,
                X_test,
                y_test,
            )
            name_map = {
                "KernelRidge": "KRR",
                "Ridge": "RR",
                "KNeighborsRegressor": "KNN",
            }
            res[
                f"{name_map.get(model.__class__.__name__, model.__class__.__name__)} - CPDM v2"
            ] = utilities

            if run_predictive:
                utilities = PredictiveBinaryDecisionMaking.inductive(
                    model,
                    self.utility_func,
                    self.threshold,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )
                res[
                    f"{name_map.get(model.__class__.__name__, model.__class__.__name__)} - PPDM"
                ] = utilities

        for model in bayesian_models:
            utilities = BDT.inductive(
                model,
                self.Decisions,
                self.utility_func,
                X_train,
                y_train,
                X_test,
                y_test,
            )
            name_map = {
                "GaussianProcessRegressor": "GPR - BDT",
                "BayesianRidge": "BRR - BDT",
            }
            res[name_map.get(model.__class__.__name__, model.__class__.__name__)] = (
                utilities
            )

        return res

    def _select_frequentist_models(
        self,
        X_train,
        y_train,
        include_knn=True,
        include_ridge=True,
        include_krr=True,
        n_splits=5,
        search_space_knn=None,
        search_space_ridge=None,
        search_space_krr=None,
        random_state=None,
    ):
        """
        Select and return frequentist models (KNN, Ridge, KRR) with tuned hyperparameters.

        This method performs cross-validated model selection for the specified models
        using the provided hyperparameter search spaces.

        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray
            Ground truth labels for training.
        include_knn, include_ridge, include_krr : bool, optional
            Flags wheather hyperparameter tuning should be performed on KNN, Ridge, and KRR.
        n_splits : int, optional
            Number of folds used for cross-validation during model selection (default is 5).
        search_space_knn, search_space_ridge, search_space_krr : dict or None, optional
            Hyperparameter grids for KNN, Ridge, and KRR model selection.
        random_state : int or None, optional
            Seed for reproducibility.

        Returns
        -------
        list of sklearn.base.BaseEstimator
            List of fitted scikit-learn models with best-found parameters.
        """
        models = []
        if include_knn:
            best_params_knn, _ = ModelSelection.model_selection(
                X_train,
                y_train,
                estimator=KNeighborsRegressor(metric="euclidean", n_jobs=-1),
                param_grid=search_space_knn,
                n_splits=n_splits,
                random_state=random_state,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
            )
            knn = KNeighborsRegressor(metric="euclidean", n_jobs=-1, **best_params_knn)
            models.append(knn)

        if include_ridge:
            best_params_ridge, _ = ModelSelection.model_selection(
                X_train,
                y_train,
                estimator=Ridge(random_state=random_state),
                param_grid=search_space_ridge,
                n_splits=n_splits,
                random_state=random_state,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
            )
            ridge = Ridge(random_state=random_state, **best_params_ridge)
            models.append(ridge)

        if include_krr:
            best_params_krr, _ = ModelSelection.model_selection(
                X_train,
                y_train,
                estimator=KernelRidge(kernel=RBF(length_scale=1.0)),
                param_grid=search_space_krr,
                n_splits=n_splits,
                random_state=random_state,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
            )
            kernel = RBF(length_scale=best_params_krr["kernel__length_scale"])
            krr = KernelRidge(kernel=kernel, alpha=best_params_krr["alpha"])
            models.append(krr)

        return models

    def _online_setting(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        run_v1=False,
        run_v2=True,
        run_predictive=False,
        include_knn=True,
        include_ridge=True,
        include_krr=True,
        n_splits=5,
        search_space_knn=None,
        search_space_ridge=None,
        search_space_krr=None,
        include_bayesian_ridge=True,
        include_gp=True,
        random_state=None,
    ):
        """
        Run the online learning pipeline using CPDM, PPDM, and BDT models.

        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray
            Ground truth labels for training.
        X_test : np.ndarray
            Feature matrix for testing.
        y_test : np.ndarray
            Ground truth labels for testing.
        run_v1, run_v2, run_predictive : bool
            Flags for enabling CPDM v1, CPDM v2, and PPDM.
        include_knn, include_ridge, include_krr : bool
            Flags for enabling frequentist models for CPDM and PPDM.
        include_bayesian_ridge, include_gp : bool
            Flags for enabling Bayesian models.
        n_splits : int
            Number of folds for model selection.
        search_space_knn, search_space_ridge, search_space_krr : dict or None
            Hyperparameter grids for model selection.
        random_state : int or None
            Seed for reproducibility.

        Returns
        -------
        dict
            Dictionary mapping method names to their utility values.
        """
        res = {}
        if include_knn:
            nnpm = NearestNeighboursPredictionMachine(k=5)
            if run_v1:
                utilities = CPDM.online_v1(
                    nnpm,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    search_space=search_space_knn,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                res["KNN - CPDM v1"] = utilities

            if run_v2:
                utilities = CPDM.online_v2(
                    nnpm,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    search_space=search_space_knn,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                res["KNN - CPDM v2"] = utilities

            if run_predictive:
                knn = KNeighborsRegressor(metric="euclidean", n_jobs=-1)
                utilities = PredictiveBinaryDecisionMaking.online(
                    knn,
                    self.utility_func,
                    self.threshold,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    param_grid=search_space_knn,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                res["KNN - PPDM"] = utilities

        if include_ridge:
            rpm = RidgePredictionMachine(a=0)
            if run_v1:
                utilities = CPDM.online_v1(
                    rpm,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    search_space=None,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                res["RR - CPDM v1"] = utilities

            if run_v2:
                utilities = CPDM.online_v2(
                    rpm,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    search_space=None,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                res["RR - CPDM v2"] = utilities

            if run_predictive:
                ridge = Ridge(random_state=random_state)
                utilities = PredictiveBinaryDecisionMaking.online(
                    ridge,
                    self.utility_func,
                    self.threshold,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    param_grid=search_space_ridge,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                res["RR - PPDM"] = utilities

        if include_krr:
            kernel = RBF(length_scale=1.0)
            krpm = KernelRidgePredictionMachine(kernel=kernel)
            if run_v1:
                utilities = CPDM.online_v1(
                    krpm,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    search_space=search_space_krr,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                res["KRR - CPDM v1"] = utilities

            if run_v2:
                utilities = CPDM.online_v2(
                    krpm,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    search_space=search_space_krr,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                res["KRR - CPDM v2"] = utilities

            if run_predictive:
                kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-10, 1e10))
                krr = KernelRidge(kernel=kernel)
                utilities = PredictiveBinaryDecisionMaking.online(
                    krr,
                    self.utility_func,
                    self.threshold,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    param_grid=search_space_krr,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                res["KRR - PPDM"] = utilities

        if include_bayesian_ridge:
            bayes_ridge = BayesianRidge()
            utilities = BDT.online(
                bayes_ridge,
                self.Decisions,
                self.utility_func,
                X_train,
                y_train,
                X_test,
                y_test,
            )
            res["BRR - BDT"] = utilities

        if include_gp:

            def custom_optimizer(obj_func, initial_theta, bounds):
                theta_opt, func_min, _ = fmin_l_bfgs_b(
                    obj_func, initial_theta, bounds=bounds, maxiter=50000
                )
                return theta_opt, func_min

            gp = GaussianProcessRegressor(
                kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-10, 1e10)),
                optimizer=custom_optimizer,
                normalize_y=True,
                random_state=random_state,
            )

            utilities = BDT.online(
                gp, self.Decisions, self.utility_func, X_train, y_train, X_test, y_test
            )
            res["GPR - BDT"] = utilities

        return res
