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
    def create_normal_df(N, F):
        X = np.random.normal(loc=0, scale=1, size=(N, F))  # Standard normal features
        beta = np.random.uniform(0, 1, size=F)  # Random weights
        raw_Y = X @ beta  # Linear combination

        # Rank transformation to spread values evenly
        Y = Y = expit(raw_Y)

        df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(F)])
        df["Target"] = Y

        return df

    @staticmethod
    def create_skewed_normal_df(N, F, skewness):
        X = np.zeros((N, F))
        skew_params = np.random.uniform(-skewness, skewness, size=F)
        for i in range(F):
            X[:, i] = stats.skewnorm.rvs(a=skew_params[i], size=N)

        # Scale X before computing Y
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # Standardize features (mean=0, std=1)

        beta = np.random.uniform(0, 1, size=F)  # Random weights
        raw_Y = X_scaled @ beta  # Linear combination

        # Rank transformation to spread values evenly
        Y = expit(raw_Y)

        df = pd.DataFrame(X_scaled, columns=[f"Feature_{i}" for i in range(F)])
        df["Target"] = Y

        return df

    @staticmethod
    def create_high_kurtosis_normal_df(N, F, scale):
        X = np.zeros((N, F))
        for i in range(F):
            X[:, i] = stats.t.rvs(df=scale, size=N)

        beta = np.random.uniform(0, 1, size=F)  # Random weights
        raw_Y = X @ beta  # Linear combination

        Y = expit(raw_Y)

        df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(F)])
        df["Target"] = Y

        return df

    @staticmethod
    def create_bimodal_normal_df(N, F, separation):
        X = np.zeros((N, F))

        for i in range(F):
            # Create bimodal features by sampling from two normal distributions
            mix = np.random.choice([0, 1], size=N, p=[0.5, 0.5])  # 50% from each mode
            X[:, i] = mix * np.random.normal(loc=-separation, scale=1, size=N) + (
                1 - mix
            ) * np.random.normal(loc=separation, scale=1, size=N)

        # Scale X before computing Y
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # Standardize features (mean=0, std=1)

        beta = np.random.uniform(0, 1, size=F)  # Random weights
        raw_Y = X_scaled @ beta  # Linear combination

        Y = expit(raw_Y)  # Sigmoid transformation for bounded [0,1] target

        df = pd.DataFrame(X_scaled, columns=[f"Feature_{i}" for i in range(F)])
        df["Target"] = Y

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
