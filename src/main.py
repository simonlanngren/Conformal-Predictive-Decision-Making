# Internal imports
from .functions.CPDM import CPDM
from .functions.BDT import BDT
from .functions.data_generation import DataGeneration
from .functions.model_selection import ModelSelection
from .functions.predictive_decision_making import PredictiveDecisionMaking

# External imports
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from itertools import product

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from skopt.space import Real, Categorical, Integer

from online_cp.CPS import RidgePredictionMachine
from online_cp.CPS import NearestNeighboursPredictionMachine


class Main:
    def __init__(self, df_params, utility_dict, subset_size, epsilon, datasplit_dict, config_dict):
        np.random.seed(2025)
        self.df_params = df_params
        self.utility_dict = utility_dict
        self.subset_size = subset_size
        self.datasplit_dict = datasplit_dict
        self.config_dict = config_dict
        self.Decisions = {0, 1}
        self.predictive_threshold = 0.5
        self.epsilon = epsilon
        
    def run(self):
        splits = self.data_generation()
        
        if self.config_dict['mode'] == "Inductive":
            models, bayesian_models = self.model_selection_and_training(self.config_dict, splits)
            plot_dict = self.inductive_setting(models, bayesian_models, splits)
        
        if self.config_dict['mode'] == "Online":
            plot_dict, decisions = self.online_setting(splits)
        
        if self.config_dict['optimal']:
            _, average_utility = CPDM.optimal_decision_making(self.Decisions, splits['y_test'], self.utility_func)
            plot_dict["Optimal"] = average_utility
            
        #print([d_r == d_k for d_r, d_k in zip(decisions["CPDM - NNPM"], decisions["CPDM - Ridge"])])
        
        # Creating the plot
        styles = ['-', '--', '-.', ':']
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'yellow']

        style_color_combinations = list(product(styles, colors))

        for i, (label, values) in enumerate(plot_dict.items()):
            x = list(range(1, len(values) + 1))
            style, color = style_color_combinations[i % len(style_color_combinations)]
            plt.plot(x, values, label=label, linestyle=style, color=color)
        
        plt.xlabel('Test Case')
        plt.ylabel('Average Utility')
        plt.legend()
        plt.title(f"Average Utility Over Test Cases - {self.config_dict['mode']} Setting")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def online_setting(self, splits):
        plot_dict = {}
        decisions = {}
        if self.config_dict['knn']:
            cps = NearestNeighboursPredictionMachine(k = 10)
            decisions_knn, average_utility, _ = (
                CPDM.online_CPDM(
                    self.Decisions,
                    splits['X_train_full'],
                    splits['y_train_full'],
                    splits['X_test'],
                    splits['y_test'],
                    self.utility_func,
                    self.epsilon,
                    cps,
                )
            )
            plot_dict["CPDM - NNPM"] = average_utility
            
            decisions["CPDM - NNPM"] = decisions_knn

            
            if self.config_dict['predictive']:
                model = KNeighborsRegressor(n_neighbors=5) # TODO: Hyperparameter tuning
                _, average_utility = (
                    PredictiveDecisionMaking.online_predictive_decision_making(
                        splits['X_train_full'],
                        splits['y_train_full'],
                        splits['X_test'],
                        splits['y_test'],
                        self.utility_func,
                        model,
                        self.predictive_threshold,
                    )
                )
                plot_dict["NNPM Predictive"] = average_utility
                
        if self.config_dict['ridge']:
            cps = RidgePredictionMachine(autotune=True)
            decisions_ridge, average_utility, _ = (
                CPDM.online_CPDM(
                    self.Decisions,
                    splits['X_train_full'],
                    splits['y_train_full'],
                    splits['X_test'],
                    splits['y_test'],
                    self.utility_func,
                    self.epsilon,
                    cps,
                )
            )
            plot_dict["CPDM - Ridge"] = average_utility
            decisions["CPDM - Ridge"] = decisions_ridge
            
            if self.config_dict['predictive']:
                model = Ridge()
                _, average_utility = (
                    PredictiveDecisionMaking.online_predictive_decision_making(
                        splits['X_train_full'],
                        splits['y_train_full'],
                        splits['X_test'],
                        splits['y_test'],
                        self.utility_func,
                        model,
                        self.predictive_threshold,
                    )
                )
                plot_dict["Ridge Predictive"] = average_utility
                
        if self.config_dict['bayesian_ridge']:
            model = BayesianRidge()
            _, average_utility = BDT.online_BDT(
                self.Decisions,
                splits['X_train_full'],
                splits['y_train_full'],
                splits['X_test'],
                splits['y_test'],
                self.utility_func,
                model,
            )            
            plot_dict["BDT - Bayesian Ridge"] = average_utility

            if self.config_dict['predictive']:
                _, average_utility = (
                    PredictiveDecisionMaking.online_predictive_decision_making(
                        splits['X_train_full'],
                        splits['y_train_full'],
                        splits['X_test'],
                        splits['y_test'],
                        self.utility_func,
                        model,
                        self.predictive_threshold,
                    )
                )
                plot_dict["Bayesian Ridge Predictive"] = average_utility
            
        if self.config_dict['gp']:
            model = gp = GaussianProcessRegressor(kernel=C(1.0) * RBF(length_scale=1.0), alpha=1e-3, normalize_y=True)
            _, average_utility = BDT.online_BDT(
                self.Decisions,
                splits['X_train_full'],
                splits['y_train_full'],
                splits['X_test'],
                splits['y_test'],
                self.utility_func,
                model,
            )            
            plot_dict["BDT - GP"] = average_utility
            
            if self.config_dict['predictive']:
                _, average_utility = (
                    PredictiveDecisionMaking.online_predictive_decision_making(
                        splits['X_train_full'],
                        splits['y_train_full'],
                        splits['X_test'],
                        splits['y_test'],
                        self.utility_func,
                        model,
                        self.predictive_threshold,
                    )
                )
                plot_dict["GP Predictive"] = average_utility
                
        return plot_dict, decisions
    
    def inductive_setting(self, models, bayesian_models, splits):
        plot_dict = {}
        for model in models:
            _, average_utility = CPDM.inductive_CPDM(
                self.Decisions,
                splits['X_train'],
                splits['y_train'],
                splits['X_cal'],
                splits['y_cal'],
                splits['X_test'],
                splits['y_test'],
                self.utility_func,
                model,
            )
            plot_dict[f"CPDM - {model.__class__.__name__}"] = average_utility
            
            if self.config_dict['predictive']:
                _, average_utility = (
                    PredictiveDecisionMaking.inductive_predictive_decision_making(
                        splits['X_train_full'],
                        splits['y_train_full'],
                        splits['X_test'],
                        splits['y_test'],
                        self.utility_func,
                        model,
                        self.predictive_threshold,
                    )
                )
                plot_dict[f"{model.__class__.__name__} Predictive"] = average_utility
                
        
        for model in bayesian_models:
            _, average_utility = BDT.inductive_BDT(
                self.Decisions,
                splits['X_train_full'],
                splits['y_train_full'],
                splits['X_test'],
                splits['y_test'],
                self.utility_func,
                model,
            )
            plot_dict[f"BDT - {model.__class__.__name__}"] = average_utility

            if self.config_dict['predictive']:
                _, average_utility = (
                    PredictiveDecisionMaking.inductive_predictive_decision_making(
                        splits['X_train_full'],
                        splits['y_train_full'],
                        splits['X_test'],
                        splits['y_test'],
                        self.utility_func,
                        model,
                        self.predictive_threshold,
                    )
                )
                plot_dict[f"{model.__class__.__name__} Predictive"] = average_utility

        return plot_dict
            
        
    def model_selection_and_training(self, config_dict, splits):
        models = []
        bayesian_models = []
        
        if self.config_dict['knn']:
            best_params_knn = self.model_selection_knn(splits)
            knn = KNeighborsRegressor(**best_params_knn)
            models.append(knn)
            
        if self.config_dict['ridge']:            
            best_params_ridge = self.model_selection_ridge(splits)
            ridge = Ridge(**best_params_ridge)
            models.append(ridge)

            
        if self.config_dict['bayesian_ridge']:
            best_params_bayes_ridge = self.model_selection_bayes_ridge(splits)
            bayes_ridge = BayesianRidge(**best_params_bayes_ridge)
            bayesian_models.append(bayes_ridge)
        
        if self.config_dict['gp']:
            gp = GaussianProcessRegressor(kernel=C(1.0) * RBF(length_scale=1.0), alpha=1e-3, normalize_y=True)
            bayesian_models.append(gp)
        
        for model in models:
            test_score = ModelSelection.evaluate(splits['X_train'], splits['y_train'], splits['X_test'], splits['y_test'], model, mean_squared_error)
            print(f"{model.__class__.__name__}: Test Score (MSE): {test_score:.3f}")
        
        for model in bayesian_models:
            test_score = ModelSelection.evaluate(splits['X_train_full'], splits['y_train_full'], splits['X_test'], splits['y_test'], model, mean_squared_error)
            print(f"{model.__class__.__name__}: Test Score (MSE): {test_score:.3f}")
        
        return models, bayesian_models
    
    
    def model_selection_bayes_ridge(self, splits):
        bayes_ridge = BayesianRidge()

        bayes_ridge_search_space = {
            'alpha_1': Real(1e-6, 1e-2, prior='log-uniform'),
            'alpha_2': Real(1e-6, 1e-2, prior='log-uniform'),
            'lambda_1': Real(1e-6, 1e-2, prior='log-uniform'),
            'lambda_2': Real(1e-6, 1e-2, prior='log-uniform')
        }

        best_params_bayes_ridge, best_score_bayes_ridge = ModelSelection.bayesian_model_selection(splits['X_train_full'], splits['y_train_full'], bayes_ridge, bayes_ridge_search_space)

        ModelSelection.print_cv_results(str(best_params_bayes_ridge), best_score_bayes_ridge)

        return best_params_bayes_ridge
            
    
    def model_selection_knn(self, splits):
        knn = KNeighborsRegressor(n_jobs=-1)

        knn_search_space = {
            'n_neighbors': Integer(1, 20),
            'weights': Categorical(['uniform', 'distance']),
            'p': Integer(1, 2)
        }

        best_params_knn, best_score_knn = ModelSelection.bayesian_model_selection(splits['X_train'], splits['y_train'], knn, knn_search_space)

        ModelSelection.print_cv_results(str(best_params_knn), best_score_knn)
        
        return best_params_knn
    
        
    def model_selection_ridge(self, splits):
        ridge = Ridge()

        ridge_search_space = {
            'alpha': Real(1e-4, 1e+4, prior='log-uniform')
        }

        best_params_ridge, best_score_ridge = ModelSelection.bayesian_model_selection(splits['X_train'], splits['y_train'], ridge, ridge_search_space)

        ModelSelection.print_cv_results(str(best_params_ridge), best_score_ridge)
        
        return best_params_ridge
        
    
    def data_generation(self):
        df = DataGeneration.generate_distribution(**self.df_params)
        
        DataGeneration.plot_histograms_and_metrics(df, "Generated Distribution")
        
        subset = df.sample(n=self.subset_size, random_state=2025) 
        X = subset.drop(columns=["Target"])
        y = subset["Target"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.datasplit_dict["train_test"], random_state=2025
        )

        X_cal, X_test, y_cal, y_test = train_test_split(
            X_test, y_test, test_size=self.datasplit_dict["test_cal"], random_state=2025
        )

        # Combine training and calibration sets
        X_train_full = pd.concat([X_train, X_cal])
        y_train_full = pd.concat([y_train, y_cal])

        # Convert to numpy arrays
        splits = {
            "X_train": X_train.to_numpy(),
            "y_train": y_train.to_numpy(),
            "X_test": X_test.to_numpy(),
            "y_test": y_test.to_numpy(),
            "X_cal": X_cal.to_numpy(),
            "y_cal": y_cal.to_numpy(),
            "X_train_full": X_train_full.to_numpy(),
            "y_train_full": y_train_full.to_numpy(),
        }

        # Histogram of the target variable
        plt.figure(figsize=(10, 6))
        sns.histplot(splits["y_train"], kde=True)
        plt.title("Histogram of the target variable")
        plt.show()

        print(f"Train set size: {len(splits['y_train'])}")
        print(f"Test set size: {len(splits['y_test'])}")
        print(f"Calibration set size: {len(splits['y_cal'])}")
        print(f"Train+Calibration set size: {len(splits['y_train'])+len(splits['y_cal'])}")

        return splits

        
    def utility_func(self, y_value, decision):
        """
        Maps y_value to a utility score for a given decision based on a dictionary input.
        utility_dict should have keys: 'tp', 'tn', 'fp', 'fn'.
        """
        y_value = int(round(y_value)) # TODO: Change to threshold

        if decision:
            if y_value:
                return self.utility_dict['tp']
            else:
                return self.utility_dict['fp']
        else:
            if y_value:
                return self.utility_dict['fn']
            else:
                return self.utility_dict['tn']
