import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from online_cp.CPS import NearestNeighboursPredictionMachine
from joblib import Parallel, delayed

class ModelSelection:
    @staticmethod
    def model_selection(X_train, y_train, estimator, param_grid, n_splits=5, random_state=2025, n_jobs=-1, verbose=0, scoring="neg_mean_squared_error"):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=n_jobs, cv=cv, verbose=0, scoring=scoring)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        return best_params, best_score
    
    # TODO update name and check parameters
    @staticmethod
    def online_KFold_knn(X_train, y_train, epsilon=0.05, k_values=[1, 3, 5, 7, 10, 12], n_jobs=-1):

        def eval_sample(cps_model, x, y, epsilon):
            tau = np.random.uniform(0, 1)
            cpd = cps_model.predict_cpd(x=x)
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            return cpd.err(Gamma=Gamma, y=y)

        def eval_k(k, splits):
            cv_errors = []
            for train_idx, val_idx in splits:
                X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
                y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

                cps_cv = NearestNeighboursPredictionMachine(k=k)
                cps_cv.learn_initial_training_set(X_train_cv, y_train_cv)

                val_errors = Parallel(n_jobs=n_jobs)(
                    delayed(eval_sample)(cps_cv, x, y, epsilon)
                    for x, y in zip(X_val_cv, y_val_cv)
                )

                cv_errors.append(np.mean(val_errors))

            return k, np.mean(cv_errors)

        # Precompute KFold splits once
        kf = KFold(n_splits=5, shuffle=True)
        splits = list(kf.split(X_train))

        # Keep only valid k-values
        k_values = [k for k in k_values if k <= len(X_train) * 0.8]

        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_k)(k, splits) for k in k_values
        )

        best_k, _ = min(results, key=lambda x: x[1])
        return best_k

    @staticmethod
    def evaluate(X_train, y_train, X_test, y_test, model, scoring_function):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = scoring_function(y_test, y_pred)
        return score
