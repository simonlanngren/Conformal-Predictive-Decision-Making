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
import scipy.stats as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from skopt.space import Real, Categorical, Integer
from checkpointer import checkpoint

from online_cp.CPS import RidgePredictionMachine
from online_cp.CPS import NearestNeighboursPredictionMachine


class Main:
    def __init__(self, df_params, utility_dict, datasplit_dict, config_dict, plot_config):
        self.df_params = df_params
        self.utility_dict = utility_dict
        self.datasplit_dict = datasplit_dict
        self.config_dict = config_dict
        self.Decisions = {0, 1}
        self.predictive_threshold = 0.5
        self.plot_config = plot_config
        
    def run(self):
        
        plot_dict = self.create_plot_dict()
        
        if (
            self.plot_config.get('average_utility_with_confidence') or
            self.plot_config.get('difference_from_optimal_with_confidence') or
            self.plot_config.get('regret_with_confidence')
        ):
            extended_plot_dict = self.create_extended_plot_dict(plot_dict=plot_dict, num_runs=self.config_dict['confidence_n_runs'])

        # Creating the plots
        if self.plot_config['average_utility']:
            self.plot_average_utility(plot_dict)

        if self.plot_config['average_utility_with_confidence']:
            self.plot_average_utility_with_confidence(extended_plot_dict)

        if self.plot_config['difference_from_optimal']:
            self.plot_difference_from_optimal(plot_dict)

        if self.plot_config['difference_from_optimal_with_confidence']:
            self.plot_difference_from_optimal_with_confidence(extended_plot_dict)
            
        if self.plot_config['regret']:
            self.plot_regret(plot_dict)
            
        if self.plot_config['regret_with_confidence']:
            self.plot_regret_with_confidence(extended_plot_dict)
                        
    def create_plot_dict(self):
        splits = self.data_generation(plot_data=True)
            
        if self.config_dict['mode'] == "Inductive":
            models, bayesian_models = self.model_selection_and_training(self.config_dict, splits)
            plot_dict = self.inductive_setting(models, bayesian_models, splits)
        
        if self.config_dict['mode'] == "Online":
            plot_dict, decisions = self.online_setting(splits)
        
        if self.config_dict['optimal']:
            _, average_utility = CPDM.optimal_decision_making(self.Decisions, splits['y_test'], self.utility_func)
            plot_dict["Optimal"] = average_utility
            
        return plot_dict
        

    def plot_regret(self, plot_dict):
        if self.config_dict['mode'] != "Online":
            print("Regret plotting is only applicable in Online mode.")
            return

        if "Optimal" not in plot_dict:
            print("Cannot compute regret: 'Optimal' values missing in plot_dict.")
            return

        optimal_utilities = np.array(plot_dict["Optimal"])
        x = np.arange(1, len(optimal_utilities) + 1)

        for i, (label, utilities) in enumerate(plot_dict.items()):
            if label == "Optimal":
                continue

            utilities = np.array(utilities)
            regret = optimal_utilities - utilities
            cumulative_regret = np.cumsum(regret)

            color = plt.cm.tab10(i % 10)
            plt.plot(x, cumulative_regret, label=label, color=color)

        plt.xlabel("Test Case")
        plt.ylabel("Cumulative Regret")
        plt.title(f"Cumulative Regret Over Time - {self.config_dict['mode']} Setting")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    
    def plot_regret_with_confidence(self, extended_plot_dict, confidence=0.95):
        if self.config_dict['mode'] != "Online":
            print("Regret plotting with confidence is only applicable in Online mode.")
            return

        if "Optimal" not in extended_plot_dict:
            print("Cannot compute regret: 'Optimal' values missing in extended_plot_dict.")
            return

        # Extract the first run of "Optimal" as the baseline
        optimal_utilities = np.array(extended_plot_dict["Optimal"][0])  # Ensure it's 1D
        x = np.arange(1, len(optimal_utilities) + 1)

        for label, utilities_list in extended_plot_dict.items():
            if label == "Optimal":
                continue

            # Convert list of utilities from multiple runs into a 2D array
            utilities_array = np.array(utilities_list)  # Shape: (num_runs, num_test_cases)
            if utilities_array.ndim == 1:  # Handle case where only one run exists
                utilities_array = utilities_array[np.newaxis, :]

            # Compute regrets and cumulative regrets
            regrets = optimal_utilities - utilities_array  # Corrected to align with regret definition
            cumulative_regrets = np.cumsum(regrets, axis=1)

            # Compute mean and confidence intervals
            mean_cumulative_regret = np.mean(cumulative_regrets, axis=0)
            std_error = st.sem(cumulative_regrets, axis=0)
            margin_of_error = std_error * st.t.ppf((1 + confidence) / 2, df=cumulative_regrets.shape[0] - 1)

            # Create a DataFrame for Seaborn
            data = pd.DataFrame({
                "Test Case": x,
                "Mean Cumulative Regret": mean_cumulative_regret,
                "Lower Bound": mean_cumulative_regret - margin_of_error,
                "Upper Bound": mean_cumulative_regret + margin_of_error
            })

            # Plot using Seaborn
            sns.lineplot(x="Test Case", y="Mean Cumulative Regret", data=data, label=label)
            plt.fill_between(
                data["Test Case"],
                data["Lower Bound"],
                data["Upper Bound"],
                alpha=0.2,
                label=f"{label} {int(confidence * 100)}% CI"
            )

        plt.xlabel("Test Case")
        plt.ylabel("Cumulative Regret")
        plt.title(f"Cumulative Regret with {int(confidence * 100)}% Confidence - {self.config_dict['mode']} Setting")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def create_extended_plot_dict(self, plot_dict, num_runs=10):
        """
        Extend plot_dict by using existing values and adding more runs to compute confidence intervals.

        Args:
            plot_dict: Dictionary containing initial results.
            num_runs: Total number of runs to average over.

        Returns:
            extended_plot_dict: Dictionary with results from multiple runs.
        """
        # Initialize extended_plot_dict with existing values
        extended_plot_dict = {key: [values] for key, values in plot_dict.items()}

        # Add additional runs to reach num_runs
        additional_runs = num_runs - 1  # Subtract 1 because plot_dict already contains one run
        for _ in range(additional_runs):
            splits = self.data_generation(plot_data=False)
            if self.config_dict['mode'] == "Online":
                run_plot_dict, _ = self.online_setting(splits)
            else:
                models, bayesian_models = self.model_selection_and_training(self.config_dict, splits)
                run_plot_dict = self.inductive_setting(models, bayesian_models, splits)

            for key, values in run_plot_dict.items():
                extended_plot_dict[key].append(values)
                
            if self.config_dict['optimal']:
                _, average_utility = CPDM.optimal_decision_making(self.Decisions, splits['y_test'], self.utility_func)
                extended_plot_dict["Optimal"].append(average_utility)

        return extended_plot_dict
        
        
    def plot_average_utility(self, plot_dict):
        for i, (label, values) in enumerate(plot_dict.items()):
            x = list(range(1, len(values) + 1))
            plt.plot(x, values, label=label, alpha=1-0.05*i)
        
        plt.xlabel('Test Case')
        plt.ylabel('Average Utility')
        plt.legend()
        plt.title(f"Average Utility Over Test Cases - {self.config_dict['mode']} Setting")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
            
    def plot_average_utility_with_confidence(self, extended_plot_dict, confidence=0.95):
        data = []

        for method, test_case_runs in extended_plot_dict.items():
            for test_case_idx, run_values in enumerate(test_case_runs):
                for utility in run_values:
                    data.append({
                        "test_case": test_case_idx + 1,
                        "method": method,
                        "utility": utility
                    })

        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))

        for method in df["method"].unique():
            method_df = df[df["method"] == method]

            if method == "Optimal":
                sns.lineplot(
                    data=method_df,
                    x="test_case",
                    y="utility",
                    label=method,
                    errorbar=None,
                    linewidth=2.5,
                    linestyle="--"
                )
            else:
                sns.lineplot(
                    data=method_df,
                    x="test_case",
                    y="utility",
                    label=method,
                    errorbar=('ci', int(confidence * 100))
                )

        plt.xlabel("Test Case")
        plt.ylabel("Average Utility")
        plt.title(f"Average Utility with Confidence Intervals - {self.config_dict['mode']} Setting")
        plt.legend(title="Method")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        
        
    def plot_difference_from_optimal(self, plot_dict):
        if "Optimal" not in plot_dict:
            print("No 'Optimal' baseline found in plot_dict.")
            return

        optimal_values = np.array(plot_dict["Optimal"])

        for i, (label, values) in enumerate(plot_dict.items()):
            if label == "Optimal":
                continue  # Skip plotting the difference for the optimal itself

            values = np.array(values)
            differences = optimal_values - values  # or values - optimal_values if you want it the other way around
            x = list(range(1, len(differences) + 1))
            color = plt.cm.tab10(i % 10)
            plt.plot(x, differences, label=label, alpha=1 - 0.05 * i, color=color)

        plt.xlabel('Test Case')
        plt.ylabel('Difference from Optimal Utility')
        plt.legend()
        plt.title(f"Difference from Optimal - {self.config_dict['mode']} Setting")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_difference_from_optimal_with_confidence(self, plot_dict):
        ## TODO
        return
                        
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
                    cps=cps,
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
                    cps=cps,
                )
            )
            plot_dict["CPDM - Ridge"] = average_utility
            decisions["CPDM - Ridge"] = decisions_ridge
            
            if self.config_dict['predictive']:
                model = Ridge() # TODO: Hyperparameter tuning in each iteration
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
            if self.config_dict['model_selection']:
                best_params_knn = self.model_selection_knn(splits)
                knn = KNeighborsRegressor(**best_params_knn)
                models.append(knn)
            else:
                models.append(KNeighborsRegressor())
            
        if self.config_dict['ridge']:   
            if self.config_dict['model_selection']:
                best_params_ridge = self.model_selection_ridge(splits)
                ridge = Ridge(**best_params_ridge)
                models.append(ridge)
            else:
                models.append(Ridge())

        if self.config_dict['bayesian_ridge']:
            bayes_ridge = BayesianRidge()
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
            
            
    def model_selection_knn(self, splits):
        knn = KNeighborsRegressor(n_jobs=-1)

        knn_search_space = self.config_dict['grid_search_space']['knn']

        best_params_knn, best_score_knn = ModelSelection.model_selection(splits['X_train'], splits['y_train'], estimator = knn, param_grid = knn_search_space)

        ModelSelection.print_cv_results(str(best_params_knn), best_score_knn)
        
        return best_params_knn
        
        
    def model_selection_ridge(self, splits):
        ridge = Ridge()
        
        ridge_search_space = self.config_dict['grid_search_space']['ridge']

        best_params_ridge, best_score_ridge = ModelSelection.model_selection(splits['X_train'], splits['y_train'], estimator = ridge, param_grid = ridge_search_space)

        ModelSelection.print_cv_results(str(best_params_ridge), best_score_ridge)
        
        return best_params_ridge
    
    
    def data_generation(self, plot_data=True):
        if self.df_params["diabetes"]:
            df = DataGeneration.generate_diabetes_data()
        else:
            df = DataGeneration.generate_distribution(self.df_params)
        
        if plot_data:
            DataGeneration.plot_histograms_and_metrics(df, self.df_params)

        X = df.drop(columns=["Target"])
        y = df["Target"]
        
        rng = np.random.default_rng()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.datasplit_dict["train_test"], random_state=rng.integers(1e9)
        )

        X_cal, X_test, y_cal, y_test = train_test_split(
            X_test, y_test, test_size=self.datasplit_dict["test_cal"], random_state=rng.integers(1e9)
        )

        # Combine training and calibration sets
        X_train_full = pd.concat([X_train, X_cal])
        y_train_full = pd.concat([y_train, y_cal])
        
        if self.df_params["diabetes"]:
            scaler = StandardScaler()
            scaler_full = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_cal = scaler.transform(X_cal)
            X_test = scaler.transform(X_test)
            X_train_full = scaler_full.fit_transform(X_train_full)
            
        # Convert to numpy arrays if needed
        splits = {
            "X_train": self.to_numpy_safe(X_train),
            "y_train": self.to_numpy_safe(y_train),
            "X_test": self.to_numpy_safe(X_test),
            "y_test": self.to_numpy_safe(y_test),
            "X_cal": self.to_numpy_safe(X_cal),
            "y_cal": self.to_numpy_safe(y_cal),
            "X_train_full": self.to_numpy_safe(X_train_full),
            "y_train_full": self.to_numpy_safe(y_train_full),
        }
                
        if plot_data:
            print(f"Train set size: {len(splits['y_train'])}")
            print(f"Test set size: {len(splits['y_test'])}")
            print(f"Calibration set size: {len(splits['y_cal'])}")
            print(f"Train+Calibration set size: {len(splits['y_train'])+len(splits['y_cal'])}")
            print("----------------------------------------")

        return splits
    
    
    def to_numpy_safe(self, x):
        return x.to_numpy() if hasattr(x, "to_numpy") else x


    def utility_func(self, y_value, decision):
        y_value = 1 if y_value >= self.utility_dict['threshold'] else 0

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
