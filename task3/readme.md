# Task 3 – Customer Segmentation Using Clustering

## Objective
This task aims to segment customers based on their age, annual income, and spending score using clustering techniques like K-Means, Agglomerative Clustering, and DBSCAN.

## Dataset
- File: `Mall_Customers.csv`
- Columns:
  - Customer_ID
  - Gender
  - Age
  - Annual Income (k$)
  - Spending Score (1-100)

## Approach
1. **Data Loading and Inspection**
   - Check for missing values and duplicates.
   - Analyze data types and summary statistics.

2. **Feature Engineering**
   - Create age and income groups.

3. **Scaling**
   - StandardScaler, MinMaxScaler, and RobustScaler options.

4. **Finding Optimal Clusters**
   - Elbow method with WCSS.
   - Silhouette score analysis.

5. **Clustering Techniques**
   - K-Means clustering.
   - Agglomerative clustering.
   - DBSCAN clustering.

6. **Dimensionality Reduction**
   - PCA and t-SNE for visualization.

7. **Visualizations**
   - Cluster plots.
   - Pair plots.
   - Heatmaps.

8. **Insights**
   - Customer segmentation.
   - Recommendations for targeted promotions.

## How to Run
1. Place `Mall_Customers.csv` in this folder.
2. Run `mall_customers_advanced_clustering.py`.
3. The output file `Mall_Customers_Advanced_Segmented.csv` will be created.
4. Explore the plots and insights generated.

## Libraries Used
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- kneed
- joblib

