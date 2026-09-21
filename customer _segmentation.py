

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset_path = Path(__file__).resolve().parent / "Dataset for Data Analytics (4).xlsx"

data = pd.read_excel(dataset_path)

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)


# ============================================================
# 2. CREATE CUSTOMER-LEVEL DATA
# ============================================================

customer_data = data.groupby("CustomerID").agg(
    TotalOrders=("OrderID", "count"),
    TotalQuantity=("Quantity", "sum"),
    TotalSpending=("TotalPrice", "sum"),
    AverageOrderValue=("TotalPrice", "mean"),
    AverageUnitPrice=("UnitPrice", "mean"),
    AverageItemsInCart=("ItemsInCart", "mean")
).reset_index()

print("\nCustomer-level data:")
print(customer_data.head())

print("\nNumber of customers:")
print(len(customer_data))


# ============================================================
# 3. SELECT FEATURES FOR CLUSTERING
# ============================================================

features = [
    "TotalOrders",
    "TotalQuantity",
    "TotalSpending",
    "AverageOrderValue",
    "AverageUnitPrice",
    "AverageItemsInCart"
]

X = customer_data[features]

print("\nFeatures used for clustering:")
print(features)


# ============================================================
# 4. STANDARDIZE THE DATA
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData standardized successfully!")


# ============================================================
# 5. ELBOW METHOD
# ============================================================

inertia = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.grid(True)
plt.savefig("elbow_method.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 6. SILHOUETTE SCORE
# ============================================================

silhouette_scores = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    silhouette_scores.append(score)

    print(
        f"K = {k}, "
        f"Silhouette Score = {score:.4f}"
    )


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")

plt.grid(True)
plt.savefig("silhouette_scores.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 7. SELECT BEST K
# ============================================================

best_k = range(2, 11)[
    silhouette_scores.index(max(silhouette_scores))
]

print("\nBest number of clusters:", best_k)


# ============================================================
# 8. APPLY K-MEANS
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

customer_data["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means clustering completed!")


# ============================================================
# 9. APPLY PCA
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

customer_data["PCA1"] = X_pca[:, 0]
customer_data["PCA2"] = X_pca[:, 1]

print("\nPCA completed!")

print(
    "\nExplained variance by PCA components:"
)

print(pca.explained_variance_ratio_)


# ============================================================
# 10. VISUALIZE CUSTOMER CLUSTERS
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    customer_data["PCA1"],
    customer_data["PCA2"],
    c=customer_data["Cluster"],
    cmap="viridis",
    s=40
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.title("Customer Segmentation using K-Means and PCA")

plt.colorbar(label="Cluster")

plt.grid(True)

plt.savefig("customer_clusters.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 11. ANALYZE CLUSTERS
# ============================================================

cluster_summary = customer_data.groupby("Cluster")[features].mean()

print("\nCluster Summary:")
print(cluster_summary)


# ============================================================
# 12. CUSTOMER PERSONAS
# ============================================================

print("\n================ CUSTOMER PERSONAS ================")

for cluster in sorted(customer_data["Cluster"].unique()):

    cluster_data = customer_data[
        customer_data["Cluster"] == cluster
    ]

    print(f"\nCluster {cluster}")
    print("--------------------------------")

    print(
        "Customers:",
        len(cluster_data)
    )

    print(
        "Average spending:",
        round(cluster_data["TotalSpending"].mean(), 2)
    )

    print(
        "Average orders:",
        round(cluster_data["TotalOrders"].mean(), 2)
    )

    print(
        "Average quantity:",
        round(cluster_data["TotalQuantity"].mean(), 2)
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

customer_data.to_excel(
    "customer_segmentation_results.xlsx",
    index=False
)

print(
    "\nResults saved as "
    "'customer_segmentation_results.xlsx'"
)

print("\nPROJECT 3 COMPLETED SUCCESSFULLY!")