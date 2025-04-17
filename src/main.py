# Internal imports
from .functions.Utility import Utility
from .functions.CPDM import CPDM
from .functions.BDT import BDT
from .functions.predictive_decision_making import PredictiveBinaryDecisionMaking
from .functions.model_selection import ModelSelection
from .functions.data_generation import DataGeneration
from .functions.plots import Plots

# External imports
from collections import defaultdict
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import fmin_l_bfgs_b

from online_cp.CPS import RidgePredictionMachine, NearestNeighboursPredictionMachine, KernelRidgePredictionMachine

from tqdm import tqdm

import numpy as np

class Main:
    def __init__(self, threshold, tp, tn, fp, fn):
        self.Decisions={0, 1}
        self.utility_func=Utility.create_utility_func(threshold, tp, tn, fp, fn)
        self.threshold=threshold
        
    def run_experiment(
        self,
        data,
        
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

        # model_selection_config
        n_splits=5,
        search_space_knn=None,
        search_space_ridge=None,
        search_space_krr=None
    ):

        if plot_distributions:
            DataGeneration.plot_target_distribution(data)
            DataGeneration.plot_first_feature_distribution(data)
        
        experiment = defaultdict(list)
        for i in tqdm(range(n_runs), desc="Processing runs"):            
            # Draw bootstrap sample
            bootstrap_data = data.sample(n=sample_size, replace=True, random_state=random_state)
            distinct_data = DataGeneration.add_epsilon(bootstrap_data)
            
            X = distinct_data.drop(columns=["Target"])
            y = distinct_data["Target"]
        
            # Split data
            X_proper, X_test, y_proper, y_test = train_test_split(X, y, test_size=test_size)
            X_train, X_cal, y_train, y_cal = train_test_split(X_proper, y_proper, test_size=cal_size)
        
            def to_numpy_safe(x):
                return x.to_numpy() if hasattr(x, "to_numpy") else x
            
            # Convert to numpy arrays if needed
            X_proper = to_numpy_safe(X_proper)
            y_proper = to_numpy_safe(y_proper)
            X_train  = to_numpy_safe(X_train)
            y_train  = to_numpy_safe(y_train)
            X_cal    = to_numpy_safe(X_cal)
            y_cal    = to_numpy_safe(y_cal)
            X_test   = to_numpy_safe(X_test)
            y_test   = to_numpy_safe(y_test)

            # Print data set sizes
            if i == 0 and print_split:
                print(f"Proper train set size: {len(y_proper)}")
                print(f"Train set size: {len(y_train)}")
                print(f"Calibration set size: {len(y_cal)}")
                print(f"Test set size: {len(y_test)}")
                if mode == "Inductive":
                    print(f"Significance and Coverage level (assuming they are the same): {CPDM.compute_significance_and_coverage(h=len(y_cal))*100}%")
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
                    random_state=random_state
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
                    random_state=random_state
                )
            else:
                raise ValueError(f"Unsupported learning mode: {mode}")
            
            # Store the experimental results
            for key, value in res.items():
                experiment[key].append(value)
            
            # Add the optimal results
            _, utilities = Utility.optimal_decision_making(self.Decisions, y_test, self.utility_func)
            experiment["Optimal"].append(utilities)
            
        if plot_average_utility:
            Plots.plot_average_utility(experiment, mode, confidence=plot_confidence)
            
        if plot_cumulative_regret:
            Plots.plot_cumulative_regret(experiment, mode, confidence=plot_confidence)
            
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
        models = []

        models = self._select_frequentist_models(
            X_train,
            y_train,
            include_knn=include_knn,
            include_ridge=include_ridge,
            include_krr=include_krr,
            n_splits=n_splits,
            search_space_knn=search_space_knn,
            search_space_ridge=search_space_ridge,
            search_space_krr=search_space_krr,
            random_state=random_state
        )

        bayesian_models = []
        if include_bayesian_ridge:
            bayes_ridge = BayesianRidge()
            bayesian_models.append(bayes_ridge)
        
        if include_gp:            
            def custom_optimizer(obj_func, initial_theta, bounds):
                theta_opt, func_min, _ = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=50000)
                return theta_opt, func_min
            
            gp = GaussianProcessRegressor(
                kernel = C(1.0, constant_value_bounds=(1e-7, 1e7)) * RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e7)),
                optimizer = custom_optimizer, 
                normalize_y = True,
                random_state = random_state
            )
            bayesian_models.append(gp)
        
        res = {}
        for model in models:
            if run_v1:
                utilities = CPDM.inductive_v1(
                    model,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_cal,
                    y_cal,
                    X_test,
                    y_test
                )
                res[f"v1 - {model.__class__.__name__}"] = utilities
                
            if run_v2:
                utilities = CPDM.inductive_v2(
                    model,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_cal,
                    y_cal,
                    X_test,
                    y_test
                )
                res[f"v2 - {model.__class__.__name__}"] = utilities
            
            if run_predictive:
                utilities = PredictiveBinaryDecisionMaking.inductive(
                    model,
                    self.utility_func,
                    self.threshold,
                    X_proper,
                    y_proper,
                    X_test,
                    y_test
                )
                res[f"{model.__class__.__name__} Pred"] = utilities
                
        for model in bayesian_models:
            if run_v1:
                utilities = BDT.inductive_v1(
                    model,
                    self.Decisions,
                    self.utility_func,
                    X_proper,
                    y_proper,
                    X_test,
                    y_test
                )
                res[f"v1 - {model.__class__.__name__}"] = utilities
                
            if run_v2:
                utilities = BDT.inductive_v2(
                    model,
                    self.Decisions,
                    self.utility_func,
                    X_proper,
                    y_proper,
                    X_test,
                    y_test
                )
                res[f"v2 - {model.__class__.__name__}"] = utilities
        
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
        random_state=None
    ):
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
                scoring="neg_mean_squared_error"
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
                scoring="neg_mean_squared_error"
            )
            ridge = Ridge(random_state=random_state, **best_params_ridge)
            models.append(ridge)
            
        if include_krr:
            best_params_krr, _ = ModelSelection.model_selection(
                X_train,
                y_train,
                estimator=KernelRidge(kernel=C(1.0, constant_value_bounds=(1e-7, 1e7)) * RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e7))),
                param_grid=search_space_krr,
                n_splits=n_splits,
                random_state=random_state,
                n_jobs=-1,
                scoring="neg_mean_squared_error"
            )
            best_kernel = C(best_params_krr['kernel__k1__constant_value']) * RBF(best_params_krr['kernel__k2__length_scale'])
            krr = KernelRidge(kernel=best_kernel, alpha=best_params_krr['alpha'])
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
        random_state=None
    ):
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
                    random_state=random_state
                )
                res["v1 - KNN"] = utilities
                
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
                    random_state=random_state
                )
                res["v2 - KNN"] = utilities

            if run_predictive:
                knn =KNeighborsRegressor(metric="euclidean", n_jobs=-1)
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
                    random_state=random_state
                )
                res["KNN Pred"] = utilities
        
        if include_ridge:
            rpm = RidgePredictionMachine(autotune=True)
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
                    random_state=random_state
                )
                res["v1 - Ridge"] = utilities
                
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
                    random_state=random_state
                )
                res["v2 - Ridge"] = utilities
            
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
                    random_state=random_state
                )
                res["Ridge Pred"] = utilities
                
        if include_krr:
            kernel = C(1.0, constant_value_bounds=(1e-7, 1e7)) * RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e7))
            krpm = KernelRidgePredictionMachine(kernel=kernel, autotune=True)
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
                    random_state=random_state
                )
                res["v1 - KRR"] = utilities
                
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
                    random_state=random_state
                )
                res["v2 - KRR"] = utilities
                
            if run_predictive:
                kernel = C(1.0, constant_value_bounds=(1e-7, 1e7)) * RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e7))
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
                    random_state=random_state
                )
                res["KRR Pred"] = utilities
                
        if include_bayesian_ridge:
            bayes_ridge = BayesianRidge()
            if run_v1:
                utilities = BDT.online_v1(
                    bayes_ridge,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )         
                res["v1 - Bayes Ridge"] = utilities
                
            if run_v2:
                utilities = BDT.online_v2(
                    bayes_ridge,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )         
                res["v2 - Bayes Ridge"] = utilities
            
        if include_gp:
            def custom_optimizer(obj_func, initial_theta, bounds):
                theta_opt, func_min, _ = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=50000)
                return theta_opt, func_min
            
            gp = GaussianProcessRegressor(
                kernel = C(1.0, constant_value_bounds=(1e-7, 1e7)) * RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e7)),
                optimizer = custom_optimizer, 
                normalize_y = True,
                random_state = random_state
            )
            
            if run_v1:
                utilities = BDT.online_v1(
                    gp,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )            
                res["v1 - GP"] = utilities
                
            if run_v2:
                utilities = BDT.online_v2(
                    gp,
                    self.Decisions,
                    self.utility_func,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )            
                res["v2 - GP"] = utilities
                
        return res
