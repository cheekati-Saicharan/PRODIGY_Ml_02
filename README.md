# PRODIGY\_DS\_02\_CustomerClustering

## Customer Segmentation using K-Means Clustering

### 🌟 Project Overview

This project implements a K-Means clustering algorithm to segment retail store customers based on their purchasing behavior. By analyzing features such as **Annual Income** and **Spending Score**, we identify distinct customer groups, enabling targeted marketing strategies and improved business insights.

This project was developed as the **second task** of the **Prodigy InfoTech Data Science Internship**.

---

### 📉 1. Problem Statement

In the competitive retail landscape, understanding customer behavior is critical for designing effective marketing and business strategies. Retailers gather large volumes of data, but much of it remains underutilized.

**Goal:** Identify distinct groups of customers based on their purchasing habits (annual income and spending score) to enable:

* Personalized marketing
* Customer loyalty strategies
* Data-driven business decision-making

**Objective:** Apply the **K-Means clustering algorithm** to group similar customers and uncover actionable insights.

---

### 📊 2. Dataset

**Source:** [Kaggle - Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

#### Dataset Features:

| Column Name              | Description                                             |
| ------------------------ | ------------------------------------------------------- |
| `CustomerID`             | Unique identifier for each customer                     |
| `Gender`                 | Customer gender (Male/Female)                           |
| `Age`                    | Customer age                                            |
| `Annual Income (k$)`     | Customer's annual income (in thousands of USD)          |
| `Spending Score (1-100)` | Mall-assigned score reflecting customer spending habits |

**Features Used for Clustering:**

* `Annual Income (k$)`
* `Spending Score (1-100)`

---

### 🧱 3. Technologies Used

* **Python** – Data processing and ML modeling
* **Pandas** – Data manipulation
* **NumPy** – Numerical computations
* **Matplotlib & Seaborn** – Visualizations
* **Scikit-learn** – Clustering and preprocessing (KMeans, StandardScaler)

---

### ⚙️ 4. Methodology

1. **Data Loading:** Read CSV and handle errors
2. **Exploration:** Head, Info, Null checks, Stats summary
3. **Feature Selection:** Choose income and spending score
4. **Scaling:** Use `StandardScaler` for normalization
5. **Determine Optimal K:** Apply **Elbow Method** (WCSS vs K)
6. **Apply K-Means:** Use optimal K=5 to train model
7. **Label Clusters:** Assign and store cluster IDs
8. **Visualization:** 2D Scatter plot colored by cluster
9. **Cluster Analysis:** Mean values grouped by cluster

---

### 📊 5. Results & Visuals

#### 📈 Elbow Method:

![Elbow Plot](elbow_plot.png)

* The "elbow" at **K=5** shows the optimal number of clusters.

#### 🌐 Customer Segments:

![Clusters Plot](customer_segments.png)

* Clear separation of customer types based on behavior

---

### 🤝 6. Interpretation of Customer Segments

| Cluster   | Characteristics                  | Strategy                                    |
| --------- | -------------------------------- | ------------------------------------------- |
| Cluster 0 | High Income, High Spenders (VIP) | Premium services, loyalty perks             |
| Cluster 1 | High Income, Low Spenders        | Targeted offers, identify needs             |
| Cluster 2 | Low Income, High Spenders        | Promotions, discounts                       |
| Cluster 3 | Low Income, Low Spenders         | Affordable pricing strategies               |
| Cluster 4 | Avg. Income & Spenders           | General campaigns, up-selling opportunities |

(Note: Your actual cluster IDs may vary. Adjust interpretation based on output.)

---

### 🚀 7. How to Run This Project

#### 🔧 Clone the Repo

```bash
git clone https://github.com/your-username/PRODIGY_DS_02_CustomerClustering.git
cd PRODIGY_DS_02_CustomerClustering
```

#### 📂 Download Dataset

Download **Mall\_Customers.csv** from:
[Kaggle Dataset Link](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

Place it in the same directory as your script.

#### 📅 Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

#### 💻 Run the Script

```bash
python customer_segmentation_project.py
```

---

### 🙌 8. Conclusion

This project effectively segments customers using unsupervised machine learning, providing a foundation for data-driven marketing. Future improvements could include:

* Adding more features (e.g., Age, Gender)
* Using PCA for dimensionality reduction
* Comparing K-Means with DBSCAN or Hierarchical Clustering

---

### 🌟 Author

**Cheekati-saicharan**
Intern @ Prodigy InfoTech
Task: PRODIGY\_DS\_02\_CustomerClustering
