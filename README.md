# Benchmarking Conformal Predictive Decision Making

This project provides a comprehensive framework for benchmarking Conformal Predictive Decision Making (CPDM) against alternative methods under both inductive and online learning settings. It includes implementations of:

- **Conformal Predictive Decision Making (CPDM)**
- **Bayesian Decision Theory (BDT)**
- **Point Predictive Decision Making (PPDM)**

The project supports synthetic data generation, utility-based evaluation, hyperparameter tuning, and visualization of results.

---

## Features

- Support for inductive and online decision-making pipelines
- Evaluation based on customizable utility functions
- Built-in model selection via cross-validation
- Generation of evaluation plots and raw experiment data
- Easily extensible for new models or decision strategies

---

## Predictive Decision Making Systems

| Abbreviation | Method                             | Description |
|--------------|-------------------------------------|-------------|
| CPDM v1      | Conformal Predictive Decision Making (per-decision CPS) | Fits a separate CPS per decision (Vovk & Bendtsen, 2018) |
| CPDM v2      | Conformal Predictive Decision Making (shared CPS)       | Shared CPS across decisions |
| BDT          | Bayesian Decision Theory            | Selects decisions based on expected utility from predictive distributions |
| PPDM         | Point Predictive Decision Making    | Uses point predictions and a decision threshold to make decisions |

---

## Supported Models

This project integrates the following machine learning models:

- **Frequentist Models (used for CPDM and PPDM)**
  - Ridge Regression (RR)
  - Kernel Ridge Regression (KRR)
  - k-Nearest Neighbors (KNN)

- **Bayesian Models (used for BDT)**
  - Bayesian Ridge Regression (BRR)
  - Gaussian Process Regression (GPR)

---

## Synthetic Datasets

This project uses synthetic datasets to simulate different types of regression problems with varying complexity and structure. These datasets are generated using standard functions from [scikit-learn](https://scikit-learn.org/), allowing for controlled benchmarking across models and decision-making methods.

The following dataset types are supported:

### `make_regression`

- **Type**: Linear regression  
- **Description**: Generates features and a target variable using a linear combination with optional Gaussian noise.  
- **Reference**: [`sklearn.datasets.make_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)

### `friedman1`

- **Type**: Nonlinear regression  
- **Description**: Introduces nonlinearity through polynomial and sine transforms of the input features.  
- **Reference**: [`sklearn.datasets.make_friedman1`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html)

### `friedman2`

- **Type**: Nonlinear regression  
- **Description**: Encodes interactions via feature multiplication and reciprocation.  
- **Reference**: [`sklearn.datasets.make_friedman2`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman2.html)

### `friedman3`

- **Type**: Nonlinear regression  
- **Description**: Generates targets based on trigonometric and multiplicative interactions among the input features.
- **Reference**: [`sklearn.datasets.make_friedman3`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman3.html)

To select a dataset, specify the `relationship` (e.g., `"friedman1"`) in your experiment’s configuration module.

---

## Running Experiments

### 1. Define Configuration Files

Each experiment is configured via a Python module in the `configs/` directory. Configs define:

- Data generation
- Utility weights
- Method flags
- Model inclusion
- Plotting preferences
- Search spaces for model selection

### 2. Run All Configured Experiments

```bash
python src/run_experiments.py
