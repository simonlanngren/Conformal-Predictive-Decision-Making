import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, make_friedman1, make_friedman2, make_friedman3
from sklearn.preprocessing import MinMaxScaler

class DataGeneration:    
    @staticmethod
    def generate_data(n_samples=10000, n_features=20, n_informative=5, relationship='make_regression', noise=0.1, random_state=None):
        if relationship == 'make_regression':
            X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative= n_informative, noise=noise, random_state=random_state)
            
        elif relationship == 'friedman1':
            X, y = make_friedman1(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
            
        elif relationship == 'friedman2':
            X, y = make_friedman2(n_samples=n_samples, noise=noise, random_state=random_state)
            
            # Friedman2 only supports 4 features
            friedman2_n_features = 4
            if n_features > friedman2_n_features:
                uninformative_features = np.random.rand(n_samples, n_features - friedman2_n_features)
                X = np.hstack((X, uninformative_features))
            
        elif relationship == 'friedman3':
            X, y = make_friedman3(n_samples=n_samples, noise=noise, random_state=random_state)
            
            # Friedman3 only supports 4 features
            friedman3_n_features = 4
            if n_features > friedman3_n_features:
                uninformative_features = np.random.rand(n_samples, n_features - friedman3_n_features)
                X = np.hstack((X, uninformative_features))
            
        else:
            raise ValueError(f"Unsupported relationship: {relationship}")

        # Min-max scale the target
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
        df['Target'] = y_scaled
        
        return df

    @staticmethod
    def plot_target_distribution(df, figsize=(12, 4), bins=30, kde=True):
        plt.figure(figsize=figsize)
        sns.histplot(data=df, x='Target', bins=bins, kde=kde, edgecolor='black')
        plt.title('Target Distribution')
        plt.xlabel('Target')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_first_feature_distribution(df, figsize=(12, 4), bins=30, kde=True):
        plt.figure(figsize=figsize)
        sns.histplot(data=df, x='Target', bins=bins, kde=kde, edgecolor='black')
        plt.title('Feature_0 Distribution')
        plt.xlabel('Feature_0')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()