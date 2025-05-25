
experiment_name = "f3_inductive_skewed_data"

utility_config = {
    'threshold': 0.5,
    'tp': 1,
    'tn': 2,
    'fp': -10,
    'fn': -2
}

method_config = {
    'mode': "Inductive",
    'run_v1': False,
    'run_v2': True,
    'run_predictive': True,
}

data_config = {
    'n_runs': 100,
    'sample_size': 100,
    'test_size': 0.2,
    'cal_size': 0.2,
    'random_state': 2025,
    'n_samples': 10000,
    'n_features': 5,
    'relationship': 'friedman3',
    'noise': 0.1,
}

models_config = {
    'include_ridge': False,
    'include_knn': True,
    'include_krr': True,
    'include_bayes_ridge': False,
    'include_gp': True
}

plot_config = {
    'plot_distributions': False,
    'print_split': False,
    'plot_average_utility': True,
    'plot_cumulative_regret': False,
    'plot_confidence': 0.95
}

model_selection_config = {
    'n_splits': 5,
    'search_space_knn': {
        'n_neighbors': [1, 3, 5, 7, 10, 15, 20],
    },
    'search_space_ridge': {
        'alpha': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    },
    'search_space_krr': {
        'alpha': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
        'kernel__length_scale': [1.0, 10, 100],
    }
}
