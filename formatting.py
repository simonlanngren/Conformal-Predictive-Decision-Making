bayesian_average_utilities = {}
for model in bayesian_models:
    _, average_utility = BDT.inductive_BDT(
        Decisions,
        X_normal_train_full,
        y_normal_train_full,
        X_normal_test,
        y_normal_test,
        utility_func,
        model,
    )
    _, p_average_utility = (
        PredictiveDecisionMaking.inductive_predictive_decision_making(
            Decisions,
            X_normal_train_full,
            y_normal_train_full,
            X_normal_test,
            y_normal_test,
            utility_func,
            model,
            threshold,
        )
    )
    bayesian_average_utilities[model.__class__.__name__] = {
        "optimal_average_utility": optimal_average_utility,
        "bayesian_average_utility": average_utility,
        "p_average_utility": p_average_utility,
    }
