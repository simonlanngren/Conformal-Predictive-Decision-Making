import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from scipy.stats import skew, kurtosis


class DataGeneration:
    @staticmethod
    def generate_distribution(
        N, 
        F, 
        skewness=0, 
        kurtosis_change=False, 
        kurtosis_scale=10,
        mixture=False, 
        bimodal_separation=4, 
        scaled=True
    ):
        X = np.zeros((N, F))

        for i in range(F):
            # Define base distribution for kurtosis
            if kurtosis_change:
                # Lower df gives heavier tails (higher kurtosis)
                base_dist = stats.t.rvs(df=kurtosis_scale, size=N)
            else:
                base_dist = np.random.normal(0, 1, size=N)

            # Introduce skewness
            if skewness != 0:
                # Skew-normal transformation preserving kurtosis
                base_dist = stats.skewnorm.rvs(a=skewness, loc=0, scale=1, size=N) * np.std(base_dist) + np.mean(base_dist)

            # Handle bimodality (mixture distribution)
            if mixture:
                # Create two modes with specified separation
                mix = np.random.choice([0, 1], size=N, p=[0.5, 0.5])
                mode_shift = bimodal_separation / 2

                # Generate modes based on skewness and kurtosis properties
                mode_1 = base_dist + np.random.normal(-mode_shift, 1, size=N)
                mode_2 = base_dist + np.random.normal(mode_shift, 1, size=N)

                # Combine into bimodal distribution
                X[:, i] = mix * mode_1 + (1 - mix) * mode_2
            else:
                X[:, i] = base_dist

        beta = np.random.uniform(0, 1, size=F)

        if scaled:
            scaler = StandardScaler()
            X_final = scaler.fit_transform(X)
        else:
            X_final = X

        raw_Y = X_final @ beta
        Y = expit(raw_Y)

        df = pd.DataFrame(X_final, columns=[f"Feature_{i}" for i in range(F)])
        df["Target"] = Y
        
        # Print parameters explicitly:
        print("Distribution created with:")
        print(f"Number of Samples: {N}")
        print(f"Number of Features: {F}")
        print(f"Skewness: {skewness}")
        print(f"Kurtosis Change: {kurtosis_change if kurtosis_change else False}")
        print(f"Bimodality: {mixture}")
        return df

    # Function to plot histograms and compute statistics

    @staticmethod
    def plot_histograms_and_metrics(df, dataset_name):
        feature_col = "Feature_2" if "Feature_2" in df.columns else "Feature_1"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram for Target
        sns.histplot(df["Target"], bins=50, kde=True, alpha=0.7, ax=axes[0])
        axes[0].set_xlabel("Target Value")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"{dataset_name} - Target Distribution")

        # Histogram for Feature
        sns.histplot(df[feature_col], bins=50, kde=True, alpha=0.7, ax=axes[1])
        axes[1].set_xlabel(f"{feature_col} Value")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"{dataset_name} - {feature_col} Distribution")

        plt.tight_layout()
        plt.show()

        # Compute and display statistics
        metrics = {
            "Mean": df[feature_col].mean(),
            "Standard Deviation": df[feature_col].std(),
            "Skewness": skew(df[feature_col]),
            "Kurtosis": kurtosis(df[feature_col]),
        }

        print(f"Metrics for {dataset_name}:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("\n" + "-" * 50 + "\n")

    @staticmethod
    def check_collinearity(X, threshold=0.9):
        """
        Checks for collinearity in a dataset by computing the correlation matrix.
        Returns True if any feature pairs have a correlation above the threshold.
        """
        corr_matrix = np.corrcoef(X, rowvar=False)  # Compute correlation matrix
        upper_triangle = np.triu(np.abs(corr_matrix), k=1)
        collinear_pairs = np.where(upper_triangle > threshold)
        return len(collinear_pairs[0]) > 0  # Returns True if collinear features exist
