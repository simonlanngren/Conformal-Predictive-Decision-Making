import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_friedman1, make_friedman2, make_friedman3
from sklearn.preprocessing import MinMaxScaler

class DataGeneration:    
    @staticmethod
    def generate_distribution(df_params):
        N = df_params['N']
        F = df_params['F']
        relationship = df_params['feature_target_relationship']
        n_informative = df_params['n_informative']
        scaler = MinMaxScaler()
        random_state = df_params['random_state']
        noise = df_params['noise']

        
        if relationship == 'make_regression':
            X, y = make_regression(n_samples=N, n_informative= n_informative, n_features=F, noise=noise, random_state=random_state)
        
        elif relationship == 'friedman1':
            X, y = make_friedman1(n_samples=N, n_features=F, noise=noise, random_state=random_state)
        
        elif relationship == 'friedman2':
            # Friedman2 only supports 4 features
            X, y = make_friedman2(n_samples=N, noise=noise, random_state=random_state)
            if F > 4:
                noise_features = np.random.rand(N, F - 4, random_state=random_state)
                X = np.hstack((X, noise_features))
        
        elif relationship == 'friedman3':
            # Friedman3 only supports 4 features
            X, y = make_friedman3(n_samples=N, noise=noise, random_state=random_state)
            if F > 4:
                noise_features = np.random.rand(N, F - 4, random_state=random_state)
                X = np.hstack((X, noise_features))
        
        else:
            raise ValueError(f"Unsupported relationship: {relationship}")

        # Min-max scale the target
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
        df['Target'] = y_scaled
        return df

    @staticmethod
    def plot_histograms_and_metrics(df, df_params):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # Target distribution
        plt.subplot(1, 2, 1)
        plt.hist(df['Target'], bins=30, edgecolor='black')
        plt.title('Target Distribution')
        plt.xlabel('Target')
        plt.ylabel('Frequency')

        # First feature distribution
        plt.subplot(1, 2, 2)
        plt.hist(df['Feature_0'], bins=30, edgecolor='black')
        plt.title('Feature_0 Distribution')
        plt.xlabel('Feature_0')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        # Extract parameters
        F = df_params.get('F')
        relationship = df_params.get('feature_target_relationship')

        # Determine number of informative features based on sklearn's definitions
        informative_lookup = {
            'make_regression': F,       # All features can be informative
            'friedman1': 5,             # Friedman1 uses 5 informative features
            'friedman2': 4,             # Friedman2 uses 4
            'friedman3': 4              # Friedman3 uses 4
        }
        informative_features = informative_lookup.get(relationship, 'Unknown')

        # Determine underlying feature distribution
        if relationship in ['friedman1', 'friedman2', 'friedman3']:
            if F > informative_features:
                feature_distribution = 'Uniform [0, 1] (plus added noise features)'
            else:
                feature_distribution = 'Uniform [0, 1]'
        elif relationship == 'make_regression':
            feature_distribution = 'Standard Normal (mean=0, std=1)'
        else:
            feature_distribution = 'Unknown'

        # Print metrics
        print("\n--- Dataset Info ---")
        print(f"Number of features: {F}")
        print(f"Number of informative features: {informative_features}")
        print(f"Underlying feature distribution: {feature_distribution}")