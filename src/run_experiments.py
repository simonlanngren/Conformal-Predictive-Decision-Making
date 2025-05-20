"""
run_experiments.py

Runs a series of predictive decision-making experiments defined by external config modules.

Each config specifies parameters for data generation, utility setup, model selection,
and plotting options. This script dynamically loads each config using importlib,
generates synthetic data, and runs the experiment using the Main class.

Authors: Simon Lanngren (simlann@chalmers.se), Martin Toremark (toremark@chalmers.se)
"""

import importlib

from src.functions.data_generation import DataGeneration
from src.main import Main

# List of config module paths to run
experiments = [
    # "configs.1_v1_vs_v2_standard.f1_v1_vs_v2_standard",
    # "configs.1_v1_vs_v2_standard.f2_v1_vs_v2_standard",
    # "configs.1_v1_vs_v2_standard.f3_v1_vs_v2_standard",
    # "configs.1_v1_vs_v2_standard.linear_v1_vs_v2_standard",
    # "configs.2_v1_vs_v2_skewed.f1_v1_vs_v2_skewed",
    # "configs.2_v1_vs_v2_skewed.f2_v1_vs_v2_skewed",
    # "configs.2_v1_vs_v2_skewed.f3_v1_vs_v2_skewed",
    # "configs.2_v1_vs_v2_skewed.linear_v1_vs_v2_skewed",
    "configs.3_online_standard.f1_online_standard",
    "configs.3_online_standard.f2_online_standard",
    "configs.3_online_standard.f3_online_standard",
    "configs.3_online_standard.linear_online_standard",
    "configs.4_online_skewed.f1_online_skewed_noise",
    "configs.4_online_skewed.f2_online_skewed_noise",
    "configs.4_online_skewed.f3_online_skewed_noise",
    "configs.4_online_skewed.linear_online_skewed_noise",
    # "configs.5_inductive_standard.f1_inductive_standard",
    # "configs.5_inductive_standard.f2_inductive_standard",
    # "configs.5_inductive_standard.f3_inductive_standard",
    # "configs.5_inductive_standard.linear_inductive_standard",
    # "configs.6_inductive_skewed.f1_inductive_skewed_noise",
    # "configs.6_inductive_skewed.f2_inductive_skewed_noise",
    # "configs.6_inductive_skewed.f3_inductive_skewed_noise",
    # "configs.6_inductive_skewed.linear_inductive_skewed_noise",
    # "configs.target_plots.friedman1_target_distribution",
    # "configs.target_plots.friedman2_target_distribution",
    # "configs.target_plots.friedman3_target_distribution",
    # "configs.target_plots.linear_target_distribution",
]

# Loop through and execute each experiment
for module in experiments:
    print(f"\nRunning experiment: {module}")

    # Dynamically import experiment-specific configuration
    config = importlib.import_module(module)

    # Generate synthetic data
    synthetic_data = DataGeneration.generate_data(
        n_samples=config.data_config["n_samples"],
        n_features=config.data_config["n_features"],
        relationship=config.data_config["relationship"],
        noise=config.data_config["noise"],
        random_state=config.data_config["random_state"],
    )

    # Set up main experiment logic with utility weights and name
    main = Main(
        threshold=config.utility_config["threshold"],
        tp=config.utility_config["tp"],
        tn=config.utility_config["tn"],
        fp=config.utility_config["fp"],
        fn=config.utility_config["fn"],
        experiment_name=config.experiment_name,
    )

    # Run the experiment with specified configs
    main.run_experiment(
        mode=config.method_config["mode"],
        run_v1=config.method_config["run_v1"],
        run_v2=config.method_config["run_v2"],
        run_predictive=config.method_config["run_predictive"],
        data=synthetic_data,
        n_runs=config.data_config["n_runs"],
        sample_size=config.data_config["sample_size"],
        test_size=config.data_config["test_size"],
        cal_size=config.data_config["cal_size"],
        random_state=config.data_config["random_state"],
        include_ridge=config.models_config["include_ridge"],
        include_knn=config.models_config["include_knn"],
        include_krr=config.models_config["include_krr"],
        include_bayesian_ridge=config.models_config["include_bayes_ridge"],
        include_gp=config.models_config["include_gp"],
        plot_distributions=config.plot_config["plot_distributions"],
        print_split=config.plot_config["print_split"],
        plot_average_utility=config.plot_config["plot_average_utility"],
        plot_cumulative_regret=config.plot_config["plot_cumulative_regret"],
        plot_confidence=config.plot_config["plot_confidence"],
        n_splits=config.model_selection_config["n_splits"],
        search_space_knn=config.model_selection_config["search_space_knn"],
        search_space_ridge=config.model_selection_config["search_space_ridge"],
        search_space_krr=config.model_selection_config["search_space_krr"],
    )
