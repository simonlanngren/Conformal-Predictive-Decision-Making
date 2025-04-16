import numpy as np
from sklearn.base import clone
from src.functions.model_selection import ModelSelection

class PredictiveBinaryDecisionMaking:
    @staticmethod
    def inductive(model, utility_func, threshold, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        decisions_made = np.where(y_pred >= threshold, 1, 0)
        utilities = [utility_func(y, d) for y, d in zip(y_test, decisions_made)]
        return utilities

    @staticmethod
    def online(model, utility_func, threshold, X_train, y_train, X_test, y_test, param_grid, n_splits=5, random_state=None):
        # Define seen examples
        X_seen = X_train
        y_seen = y_train
        
        decisions_made = []
        for i, x in enumerate(X_test):
            # Clone the model to avoid contamination
            model_i = clone(model)
            
            # Perform model selection
            best_params, _ = ModelSelection.model_selection(
                X_seen, y_seen, model_i, param_grid, n_splits=n_splits, random_state=random_state
            )
            
            # Clone again and set best parameters
            model_i = clone(model)
            model_i.set_params(**best_params)
            
            # Train the selected model
            model_i.fit(X_seen, y_seen)
            
            # Make predictions and decisions
            y_pred = model_i.predict(x.reshape(1, -1))
            decisions_made.append(np.where(y_pred >= threshold, 1, 0))
            
            # Update seen examples
            X_seen = np.vstack((X_seen, x))
            y_seen = np.hstack((y_seen, y_test[i]))

        # Compute utility of the decisions made
        utilities = [utility_func(y, d) for y, d in zip(y_test, decisions_made)]
        
        return  utilities
