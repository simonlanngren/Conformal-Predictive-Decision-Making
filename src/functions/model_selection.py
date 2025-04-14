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
    
    @staticmethod
    def online_cpdm_model_selection_knn(X_train, y_train, search_space, n_splits=5, random_state=None):
        def eval_sample(cps_model, x, y, epsilon=0.05):
            tau = np.random.uniform(0, 1)
            cpd = cps_model.predict_cpd(x=x)
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            return cpd.err(Gamma=Gamma, y=y)

        def eval_k(k, kf, X, y):
            cv_errors = []
            for train_idx, val_idx in kf.split(X):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                cps = NearestNeighboursPredictionMachine(k=k)
                cps.learn_initial_training_set(X_train, y_train)

                val_errors = Parallel(n_jobs=-1)(
                    delayed(eval_sample)(cps, x, y)
                    for x, y in zip(X_val, y_val)
                )

                cv_errors.append(np.mean(val_errors))

            return k, np.mean(cv_errors)

        # Setup
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        valid_ks = [k for k in search_space["n_neighbors"] if k <= len(X_train)*(1 - 1/n_splits)]

        results = Parallel(n_jobs=-1)(
            delayed(eval_k)(k, kf, X_train, y_train) for k in valid_ks
        )
        
        best_k, _ = min(results, key=lambda x: x[1])
        
        return best_k

    @staticmethod
    def evaluate(X_train, y_train, X_test, y_test, model, scoring_function):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = scoring_function(y_test, y_pred)
        return score
