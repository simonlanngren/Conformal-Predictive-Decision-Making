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

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from skopt.space import Real, Categorical, Integer

from online_cp.CPS import RidgePredictionMachine
from online_cp.CPS import NearestNeighboursPredictionMachine


class Main:
    def __init__(self, df_params, utility_dict, subset_size, datasplit_dict):
        np.random.seed(2025)
        self.df_params = df_params
        self.utility_dict = utility_dict
        self.subset_size = subset_size
        self.datasplit_dict = datasplit_dict
        
    def run(self):
        data_splits = self.data_generation()
        
        models, bayesian_models = self.model_selection_and_training(**data_splits)
        
        
    def model_selection_and_training(self, splits):
        best_params_knn = self.model_selection_knn(splits)
        best_params_ridge = self.model_selection_ridge(splits)
        best_params_bayes_ridge = self.model_selection_bayes_ridge(splits)
        
        knn = KNeighborsRegressor(**best_params_knn)
        ridge = Ridge(**best_params_ridge)
        bayes_ridge = BayesianRidge(**best_params_bayes_ridge)
        gp = GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)))
        
        knn.fit(X_train, y_train)
        ridge.fit(X_train, y_train)
        bayes_ridge.fit(X_train_full, y_train_full)
        gp.fit(X_train_full, y_train_full)
        
        models = [knn, ridge]
        
        for model in models:
            test_score = ModelSelection.evaluate(X_train, y_train, X_test, y_test, model, mean_squared_error)
            print(f"{model.__class__.__name__}: Test Score (MSE): {test_score:.3f}")
        
        bayesian_models = [bayes_ridge, gp]
        
        for model in bayesian_models:
            test_score = ModelSelection.evaluate(X_train_full, y_train_full, X_test, y_test, model, mean_squared_error)
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

        best_params_bayes_ridge, best_score_bayes_ridge = ModelSelection.bayesian_model_selection(splits["X_train_full"], y_train_full, bayes_ridge, bayes_ridge_search_space)

        ModelSelection.print_cv_results(str(best_params_bayes_ridge), best_score_bayes_ridge)

        return best_params_bayes_ridge
            
    
    def model_selection_knn(self):
        knn = KNeighborsRegressor(n_jobs=-1)

        knn_search_space = {
            'n_neighbors': Integer(1, 20),
            'weights': Categorical(['uniform', 'distance']),
            'p': Integer(1, 2)
        }

        best_params_knn, best_score_knn = ModelSelection.bayesian_model_selection(X_train, y_train, knn, knn_search_space)

        ModelSelection.print_cv_results(str(best_params_knn), best_score_knn)
        
        return best_params_knn
    
        
    def model_selection_ridge(self):
        ridge = Ridge()

        ridge_search_space = {
            'alpha': Real(1e-4, 1e+4, prior='log-uniform')
        }

        best_params_ridge, best_score_ridge = ModelSelection.bayesian_model_selection(X_train, y_train, ridge, ridge_search_space)

        ModelSelection.print_cv_results(str(best_params_ridge), best_score_ridge)
        
        return best_params_ridge
        
    
    def data_generation(self):
        df = DataGeneration.generate_distribution(**self.df_params)
        
        DataGeneration.plot_histograms_and_metrics(df, "Generated Distribution")
        
        subset = df.sample(n=self.subset_size, random_state=2025) 
        X = subset.drop(columns=["y"])
        y = subset["y"]
        
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

        return splits

        
    def utility_func(self, y_value, decision):
        """
        Maps y_value to a utility score for a given decision based on a dictionary input.
        utility_dict should have keys: 'tp', 'tn', 'fp', 'fn'.
        """
        y_value = int(round(y_value))

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
