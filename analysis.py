import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# --------------------------
# Step 1: Load the Dataset
# --------------------------
# Replace "medical_data.csv" with the path to your CSV file.
data = pd.read_csv("medical_data.csv")
print("Dataset loaded successfully. Here are the first 5 rows:")
print(data.head())

# --------------------------
# Step 2: Descriptive Statistics
# --------------------------
print("\nDescriptive Statistics for Numeric Columns:")
print(data.describe())

print("\nDescriptive Statistics for Categorical Columns:")
print(data.describe(include=['object']))

# --------------------------
# Step 3: Analysis & Visualizations for Each Column
# --------------------------
# Identify numeric and categorical columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Plot histograms and KDE for numeric columns
for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], bins=30, kde=True, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Plot count plots for categorical columns
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    order = data[col].value_counts().index
    sns.countplot(y=col, data=data, order=order, palette="viridis")
    plt.title(f"Value Counts for {col}")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# --------------------------
# Step 4: Correlation Analysis (for numeric columns)
# --------------------------
if len(numeric_columns) > 1:
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.show()
else:
    print("Not enough numeric columns for a correlation heatmap.")

# --------------------------
# Step 5: KL Divergence Analysis (Optional)
# --------------------------
# Here, we compute KL divergence for each numeric column against a reference normal distribution.
# This step is optional and assumes you want to compare each column's distribution to a normal distribution.
for col in numeric_columns:
    # Skip if there are too few unique values
    if data[col].nunique() < 10:
        continue
    mu, sigma = data[col].mean(), data[col].std()
    # Create a reference distribution: generate synthetic samples from a normal distribution with same mean and std.
    reference_data = np.random.normal(mu, sigma, size=len(data))
    
    # Create common bins using both datasets
    combined = np.concatenate([data[col].values, reference_data])
    bins = np.histogram_bin_edges(combined, bins='auto')
    
    # Calculate histograms for both distributions (density=True to get probability densities)
    real_hist, _ = np.histogram(data[col], bins=bins, density=True)
    ref_hist, _ = np.histogram(reference_data, bins=bins, density=True)
    
    # Add epsilon to avoid zero values
    epsilon = 1e-10
    real_hist += epsilon
    ref_hist += epsilon
    
    # Normalize histograms
    real_hist /= real_hist.sum()
    ref_hist /= ref_hist.sum()
    
    # Compute KL Divergence: lower value means distributions are more similar.
    kl_div = entropy(real_hist, ref_hist)
    print(f"KL Divergence for {col}: {kl_div:.4f}")
