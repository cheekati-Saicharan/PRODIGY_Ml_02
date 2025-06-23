import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os # To check file existence

# Suppress all warnings for cleaner output in a professional setting.
# In a development environment, you might temporarily disable this to see important warnings.
warnings.filterwarnings('ignore')

# --- 2. Load the Dataset ---
# Define the dataset file name
DATASET_FILE = "Mall_Customers.csv"

# Ensure 'Mall_Customers.csv' is in the same directory as this script,
# or provide the full path to the file.
if not os.path.exists(DATASET_FILE):
    print(f"Error: '{DATASET_FILE}' not found in the current directory ({os.getcwd()}).")
    print("Please download the dataset from: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python")
    print("And place it in the same directory as this Python script.")
    exit() # Exit the script if the dataset isn't found
else:
    df = pd.read_csv(DATASET_FILE)
    print("Dataset loaded successfully.")

# --- 3. Initial Data Exploration and Preprocessing ---
print("\n--- Dataset Head (First 5 Rows) ---")
print(df.head())

print("\n--- Dataset Information ---")
df.info()

print("\n--- Missing Values Check ---")
print(df.isnull().sum())
# This dataset is typically clean with no missing values.
# If there were, strategies like imputation (mean, median, mode) or removal would be applied here.

print("\n--- Dataset Shape ---")
print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

print("\n--- Basic Statistical Summary ---")
print(df.describe())

# --- 4. Feature Selection for Clustering ---
# We select 'Annual Income (k$)' and 'Spending Score (1-100)' for clustering.
# These two features are often effective in revealing distinct customer segments based on purchasing power and habits.
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print(f"\nFeatures selected for clustering (First 5 rows of X):\n{X.head()}")

# --- 5. Feature Scaling ---
# Scaling features is crucial for K-Means as it relies on Euclidean distance calculations.
# Features with larger numerical ranges (like Annual Income) would disproportionately influence distances
# without scaling. StandardScaler transforms data to have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Scaled Features (First 5 rows) ---")
print(X_scaled[:5])

# --- 6. Determine Optimal Number of Clusters (K) using the Elbow Method ---
# The Elbow Method aims to find the point where the decrease in WCSS (Within-Cluster Sum of Squares)
# begins to level off, suggesting an optimal 'k'.
wcss = [] # List to store WCSS for each K
max_k = 10 # Maximum number of clusters to test

for i in range(1, max_k + 1):
    # Initialize KMeans with n_init=10 to run the algorithm 10 times with different centroid seeds
    # and choose the best result (to avoid local optima). random_state ensures reproducibility.
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) # inertia_ attribute gives the WCSS value

# Plotting the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--', color='blue', markeredgecolor='black')
plt.title('Elbow Method for Optimal K', fontsize=18)
plt.xlabel('Number of Clusters (K)', fontsize=14)
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(range(1, max_k + 1))
plt.tight_layout()
plt.show()

print("\n--- Elbow Method Analysis ---")
print("Observe the plot above. The 'elbow' point signifies where adding more clusters no longer significantly")
print("reduces the WCSS. For this dataset, a prominent elbow is typically observed around K=5.")

# --- 7. Apply K-Means Clustering with the Optimal K ---
# Based on the Elbow Method's visual indication, optimal_k is chosen as 5.
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

# Fit KMeans and predict cluster labels for each customer
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n--- K-Means Clustering Applied with K = {optimal_k} ---")
print(f"Distribution of customers across clusters:\n{df['Cluster'].value_counts().sort_index()}")

# --- 8. Visualize the Clusters ---
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',        # Color points by the assigned cluster label
    palette='viridis',    # A vibrant color palette for distinction
    data=df,              # Use the DataFrame with the new 'Cluster' column
    s=120,                # Size of the markers
    alpha=0.8,            # Transparency of the markers
    edgecolor='black'     # Black edge for better visibility of individual points
)
# Plotting cluster centers for better visual interpretation
# cluster_centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)
# plt.scatter(cluster_centers_original_scale[:, 0], cluster_centers_original_scale[:, 1],
#             marker='X', s=300, color='red', label='Cluster Centers', edgecolor='white', linewidth=1.5)

plt.title(f'Customer Segments using K-Means Clustering (K={optimal_k})', fontsize=20, pad=20)
plt.xlabel('Annual Income (k$)', fontsize=15)
plt.ylabel('Spending Score (1-100)', fontsize=15)
plt.legend(title='Customer Cluster', title_fontsize='13', fontsize='12', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to prevent legend from overlapping plot
plt.show()

# --- 9. Analyze and Interpret Clusters in Detail ---
print("\n--- Detailed Cluster Analysis ---")
# Group by cluster and calculate mean for all relevant original features
cluster_summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)', 'Age', 'Gender']].mean().round(2)
print(cluster_summary)

print("\n--- Interpretation of Customer Segments ---")
# Providing detailed descriptions for each cluster based on typical outcomes for this dataset
# (Note: Cluster numbers are arbitrary, focus on their characteristics)

# Identify the characteristics of each cluster based on the 'cluster_summary'
# This mapping needs to be dynamic or based on visual inspection.
# For K=5, common interpretations are:

# Example mapping based on typical results for this dataset:
# Let's assume (based on typical results and the summary output)
# Cluster 0: Often Moderate Income, Moderate Spenders
# Cluster 1: Often High Income, Low Spenders
# Cluster 2: Often Low Income, High Spenders
# Cluster 3: Often Low Income, Low Spenders
# Cluster 4: Often High Income, High Spenders

# You would map these based on the actual `cluster_summary` output.
# Here's a general interpretation that assumes this typical mapping:

print("\nBased on the analysis of annual income and spending score, customers are segmented into the following distinct groups:")

# Iterating through the cluster_summary to dynamically describe each cluster
for cluster_id in cluster_summary.index:
    income = cluster_summary.loc[cluster_id, 'Annual Income (k$)']
    spending = cluster_summary.loc[cluster_id, 'Spending Score (1-100)']
    age = cluster_summary.loc[cluster_id, 'Age']
    gender_mean = cluster_summary.loc[cluster_id, 'Gender'] # 0 for Female, 1 for Male if encoded
    
    # For Gender, it's better to get value counts per cluster
    gender_dist = df[df['Cluster'] == cluster_id]['Gender'].value_counts(normalize=True)
    gender_info = ""
    if 'Male' in gender_dist and 'Female' in gender_dist:
        if gender_dist['Male'] > gender_dist['Female']:
            gender_info = f"Predominantly Male ({gender_dist['Male']:.1%})"
        else:
            gender_info = f"Predominantly Female ({gender_dist['Female']:.1%})"
    elif 'Male' in gender_dist:
        gender_info = "All Male"
    elif 'Female' in gender_dist:
        gender_info = "All Female"

    description = ""
    rationale = ""

    # These thresholds (70, 40, 60) are examples and might need slight adjustment
    # based on the specific distribution of your data if the clusters don't align perfectly.
    if income >= 70 and spending >= 70:
        description = "High Income, High Spenders (VIP Customers)"
        rationale = "These customers have both high annual income and high spending scores. They are likely the most valuable segment, often interested in premium products and exclusive offers. They represent a key target for loyalty programs."
    elif income >= 70 and spending < 40:
        description = "High Income, Low Spenders (Careful Spenders)"
        rationale = "This segment earns well but spends relatively little at the mall. They might be saving, spending elsewhere, or only visiting for specific needs. Marketing could focus on increasing their engagement and showing value."
    elif income < 40 and spending >= 60:
        description = "Low Income, High Spenders (Impulsive/Value Seekers)"
        rationale = "Despite lower income, these customers have high spending scores, indicating impulsive buying habits or a focus on value/necessity purchases. They might respond well to discounts, sales, and essential product promotions."
    elif income < 40 and spending < 40:
        description = "Low Income, Low Spenders (Budget-Conscious)"
        rationale = "This group has lower income and lower spending scores. They are likely very budget-conscious. Marketing strategies should focus on essential goods, clearance sales, and affordable options."
    else:
        description = "Average Income, Average Spenders (General Customer Base)"
        rationale = "This segment falls in the middle for both income and spending. They represent the general customer base and can be targeted with a broad range of products and promotions."

    print(f"\nCluster {cluster_id}: {description}")
    print(f"  - Avg. Annual Income: ${income:.2f}k")
    print(f"  - Avg. Spending Score: {spending:.2f}")
    print(f"  - Avg. Age: {age:.0f} years")
    print(f"  - Gender Distribution: {gender_info}")
    print(f"  - Rationale/Targeting: {rationale}")


print("\n--- Project Completed ---")
