from sklearn.model_selection import GridSearchCV, KFold
from skopt import BayesSearchCV
from checkpointer import checkpoint

class ModelSelection:
    @checkpoint
    @staticmethod
    def model_selection(
        X_train,
        y_train,
        estimator,
        param_grid,
        n_splits=5,
        random_state=43,
        n_jobs=-1,
        verbose=2,
        scoring="neg_mean_squared_error",
    ):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
            scoring=scoring,
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        return best_params, best_score

    @checkpoint
    @staticmethod
    def bayesian_model_selection(
        X_train,
        y_train,
        estimator,
        search_space,
        n_splits=5,
        random_state=43,
        n_jobs=-1,
        n_points=1,
        verbose=2,
        scoring="neg_mean_squared_error",
        n_iter=100,
    ):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        opt = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_space,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            n_points=n_points,
            cv=cv,
            random_state=random_state,
        )
        opt.fit(X_train, y_train)

        best_params = opt.best_params_
        best_score = opt.best_score_

        return best_params, best_score

    @staticmethod
    def print_cv_results(best_params, best_score):
        print(f"Best Parameters: {best_params}")
        print(f"Best CV Score: {best_score:.3f}")

    @staticmethod
    def evaluate(X_train, y_train, X_test, y_test, model, scoring_function):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = scoring_function(y_test, y_pred)
        return score
