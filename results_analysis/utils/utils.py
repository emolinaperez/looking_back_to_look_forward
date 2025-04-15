import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import davies_bouldin_score
import math
from scipy import stats
import logging


class FeatureExtractor:
    """
    A class for extracting features from time series data.
    """

    def get_final_value(self, df, feature):
        """
        Retrieve the final value of a specified feature from a DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature (str): The column name of the feature to retrieve the final value from.

        Returns:
        The final value of the specified feature in the DataFrame.
        """
        return df[feature].iloc[-1]
    
    def get_max_value(self, df, feature):
        """
        Get the maximum value of a specified feature in a DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature (str): The column name of the feature for which to find the maximum value.

        Returns:
        float or int: The maximum value of the specified feature.
        """
        return df[feature].max()
    
    def get_min_value(self, df, feature):
        """
        Get the minimum value of a specified feature in a DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature (str): The name of the feature/column for which to find the minimum value.

        Returns:
        float: The minimum value of the specified feature in the DataFrame.
        """
        return df[feature].min()
    
    def get_auc(self, df, feature):
        """
        Calculate the area under the curve (AUC) for a given feature over time.

        Parameters:
        df (pandas.DataFrame): DataFrame containing the data.
        feature (str): The name of the feature column for which to calculate the AUC.

        Returns:
        float: The calculated AUC value.
        """
        return np.trapezoid(df[feature], df["time"])
    
    def get_time_to_collapse(self, df, feature, threshold):
        """
        Calculate the time at which a specified feature in the DataFrame falls below a given threshold.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature (str): The column name of the feature to be evaluated.
        threshold (float): The threshold value to determine collapse.

        Returns:
        pandas.Timestamp or scalar: The time at which the feature first falls below the threshold.
                                    If the feature never falls below the threshold, returns the maximum time in the DataFrame.
        """
        collapse_times = df[df[feature] < threshold]["time"]
        return collapse_times.iloc[0] if not collapse_times.empty else df["time"].max()

    def get_std(self, df, feature):
        """
        Calculate the standard deviation of a specified feature in a DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature (str): The name of the feature/column for which to calculate the standard deviation.

        Returns:
        float: The standard deviation of the specified feature.
        """
        return df[feature].std()
    
    def get_growth_rate(self, df, feature, t0, t1):
        """
        Calculate the growth rate of a specified feature between two time points.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        feature (str): The feature/column name for which the growth rate is to be calculated.
        t0 (int or float): The initial time point.
        t1 (int or float): The final time point.

        Returns:
        float: The growth rate of the feature between time t0 and t1. Returns NaN if the values at t0 or t1 are not available.
        """
        try:
            val_t0 = df.loc[df["time"] == t0, feature].values[0]
            val_t1 = df.loc[df["time"] == t1, feature].values[0]
        except IndexError:
            val_t0, val_t1 = np.nan, np.nan
        return (val_t1 - val_t0) / (t1 - t0) if pd.notna(val_t1) else np.nan
    
    def calculate_delta(self, df, feature, t):
        """
        Calculate the delta of a specified feature at a given time point.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature (str): The name of the feature/column for which to calculate the delta.
        t (int or float): The time point at which to calculate the delta.

        Returns:
        float: The delta value of the specified feature at time t.
        """
        return df.loc[df["time"] == t, feature].values[0] - df.loc[df["time"] == (t - 1), feature].values[0]
    
    def get_max_value_at_time_window(self, df, feature, t0, t1):
        """
        Get the maximum value of a specified feature within a time window.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature (str): The name of the feature/column for which to find the maximum value.
        t0 (int or float): The start time of the window.
        t1 (int or float): The end time of the window.

        Returns:
        float: The maximum value of the specified feature within the time window.
        """
        return df[(df["time"] >= t0) & (df["time"] <= t1)][feature].max()
    
    def get_min_value_at_time_window(self, df, feature, t0, t1):
        """
        Get the minimum value of a specified feature within a time window.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature (str): The name of the feature/column for which to find the minimum value.
        t0 (int or float): The start time of the window.
        t1 (int or float): The end time of the window.

        Returns:
        float: The minimum value of the specified feature within the time window.
        """
        return df[(df["time"] >= t0) & (df["time"] <= t1)][feature].min()
       
    def extract_ts_features(self, group_df, collapse_thresh=0.01):
        """
        Extracts time series features from a given DataFrame for specified state variables.
        Adds additional delta features for the first 20 periods.
        """
        features = {}
        state_vars = ["Resources", "Economy", "Bureaucracy", "Pollution"]

        # Ensure the data is sorted by time
        group_df = group_df.sort_values("time")

        for state_var in state_vars:
            # Standard features
            features[state_var + "_final"] = self.get_final_value(group_df, state_var)
            features[state_var + "_max"] = self.get_max_value(group_df, state_var)
            features[state_var + "_min"] = self.get_min_value(group_df, state_var)
            features[state_var + "_auc"] = self.get_auc(group_df, state_var)
            
            # features[state_var + "_max_min_diff"] = features[state_var + "_max"] - features[state_var + "_min"]
            # features[state_var + "_final_min_diff"] = features[state_var + "_final"] - features[state_var + "_min"]
            # features[state_var + "_final_max_diff"] = features[state_var + "_final"] - features[state_var + "_max"]
            # features[state_var + "initial_min_diff"] = group_df[state_var].iloc[0] - features[state_var + "_min"]
            # features[state_var + "_initial_max_diff"] = group_df[state_var].iloc[0] - features[state_var + "_max"]
            features[state_var + "_final_initial_diff"] = features[state_var + "_final"] - group_df[state_var].iloc[0]


            # features[state_var + "_time_to_collapse"] = self.get_time_to_collapse(group_df, state_var, collapse_thresh)
            # features[state_var + "_std"] = self.get_std(group_df, state_var)
            # features[state_var + "_growth_rate_t4"] = self.get_growth_rate(group_df, state_var, 0, 4)
            # features[state_var + "_growth_rate_t25"] = self.get_growth_rate(group_df, state_var, 0, 25)
            
        
            # # Early delta: difference between the 5th period and the initial value
            # early_delta = group_df[state_var].iloc[5] - group_df[state_var].iloc[0]
            # features[state_var + "_early_delta"] = early_delta
            # # Mid delta: difference between the 15th period and the initial value
            # mid_delta = group_df[state_var].iloc[20] - group_df[state_var].iloc[0]
            # features[state_var + "_mid_delta"] = mid_delta
            
            # # Early average difference: average change between consecutive values over the first 5 periods
            # early_avg_diff = group_df[state_var].iloc[:5].diff().mean()
            # features[state_var + "_early_avg_diff"] = early_avg_diff

            # # Mid average difference: average change between consecutive values over the first 20 periods
            # mid_avg_diff = group_df[state_var].iloc[:20].diff().mean()
            # features[state_var + "_mid_avg_diff"] = mid_avg_diff

            # # Calculate deltas for the first 25 periods every 5 periods
            # for i in range(5, 31, 5):
            #     features[state_var + f"_delta_{i}"] = self.calculate_delta(group_df, state_var, i)
            
            # Calculate the delta for the after the 25th period every 25 periods
            for i in range(25, 201, 25):
                features[state_var + f"_delta_{i}"] = self.calculate_delta(group_df, state_var, i)

            # # Calculate the maximum value of the feature every 25 periods
            # for i in range(0, 201, 25):
            #     features[state_var + f"_max_{i}"] = self.get_max_value_at_time_window(group_df, state_var, i, i+25)

            # #TODO Calculate the minimum value of the feature every 25 periods
            # for i in range(0, 201, 25):
            #     features[state_var + f"_min_{i}"] = self.get_min_value_at_time_window(group_df, state_var, i, i+25)

        return pd.Series(features)

# Example usage:
# Assuming output_df is the DataFrame containing the simulation results with columns:
#   Resources, Economy, Bureaucracy, Pollution, time, run_id
#
# feature_extractor = FeatureExtractor()
# features_df = output_df.groupby("run_id").apply(feature_extractor.extract_ts_features).reset_index()

class EDAUtils:

    @staticmethod
    def get_skewed_variables(df, threshold=1):
        """
        Identify skewed variables in the DataFrame based on a specified threshold.
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            threshold (float): The threshold value for skewness.
        Returns:
            list: A list of column names that are skewed.
        """
        # Check if the DataFrame is empty
        if df.empty:
            print("DataFrame is empty.")
            return []
        # Check if the DataFrame has numeric columns
        if df.select_dtypes(include=[np.number]).empty:
            print("No numeric columns in DataFrame.")
            return []
     
        # Calculate skewness for numeric features in the DataFrame
        skewness = df.skew(numeric_only=True)
    
        skewed_features = skewness[abs(skewness) > threshold].index.tolist()

        return skewed_features

    @staticmethod
    def apply_log_transform(df, cols):
        """
        Applies log transformation to the specified columns of a DataFrame.
        If any values in a column are <= 0, an offset is added to ensure all values are positive.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            cols (list of str): List of column names to transform.

        Returns:
            pd.DataFrame: A new DataFrame with log-transformed columns.
        """
        df_trans = df.copy()
        for col in cols:
            # Check if the column contains non-positive values.
            if (df_trans[col] <= 0).any():
                # Offset by adding (abs(min_value) + 1) so that all values become positive.
                offset = abs(df_trans[col].min()) + 1
                df_trans[col] = np.log1p(df_trans[col] + offset)
                print(f"Applied log1p with offset {offset} to column: {col}")
            else:
                df_trans[col] = np.log(df_trans[col])
                print(f"Applied natural log to column: {col}")
        
        return df_trans
    
    @staticmethod
    def plot_histograms(df, figsize=(20, 15)):
        # Histograms for each numeric feature
        df.hist(figsize=figsize)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_corr_heatmap(df, figsize=(20, 15)):
        # Correlation heatmap to inspect relationships between features
        plt.figure(figsize=figsize)
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.show()

class ClusteringPipeline:
    
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info("ClusteringPipeline initialized.")

    def drop_features(self, df, features_to_drop):
        """
        Drop specified features from the DataFrame.
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            features_to_drop (list): List of features to drop.
        Returns:
            pd.DataFrame: A DataFrame with specified features dropped.
        """
        if df.empty:
            self.logger.warning("DataFrame is empty.")
            return pd.DataFrame()
        self.logger.info(f"Dropping features: {features_to_drop}")
        df_dropped = df.drop(columns=features_to_drop, errors='ignore')
        return df_dropped

    def scale_features(self, df):
        """
        Scale features using StandardScaler.
        Parameters:
            df (pd.DataFrame): The input DataFrame containing features to be scaled.
        Returns:
            pd.DataFrame: A DataFrame with scaled features.
        """
        if df.empty:
            self.logger.warning("DataFrame is empty.")
            return pd.DataFrame()
        if df.select_dtypes(include=[np.number]).empty:
            self.logger.warning("No numeric columns in DataFrame.")
            return pd.DataFrame()
        self.logger.info("Scaling features using StandardScaler.")
        X = df.copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.logger.info("Scaled features shape: %s", X_scaled.shape)
        return X_scaled
    
    def apply_pca(self, X, n_components=0.9):
        """
        Apply PCA to reduce dimensionality of the dataset.

        Parameters:
            X (np.ndarray or pd.DataFrame): The input data to apply PCA on.
            n_components (float or int): Number of components to keep or the variance ratio to retain.

        Returns:
            np.ndarray: Transformed data after applying PCA.
        """
        # Retain enough components to explain the specified variance
        pca = PCA(n_components=n_components, random_state=42)
        df_pca = pca.fit_transform(X)
        self.logger.info("Number of components selected: %d", pca.n_components_)
        self.logger.info("New shape after PCA: %s", df_pca.shape)

        # Plot cumulative explained variance
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Explained Variance by PCA Components")
        plt.grid(True)
        plt.show()

        return df_pca
    
    def run_data_preprocessing(self, df, features_to_drop):
        """
        Run data preprocessing steps: drop features and scale.
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            features_to_drop (list): List of features to drop.
        Returns:
            pd.DataFrame: A DataFrame with scaled features.
        """
        self.logger.info("Running data preprocessing.")
        df_dropped = self.drop_features(df, features_to_drop)
        X_scaled = self.scale_features(df_dropped)
        X_pca = self.apply_pca(X_scaled)
        self.logger.info("Data preprocessing completed.")
        return X_pca
    
    def kmeans_elbow_plot(self, df, k_range=range(2, 12)):
        """
        Generate the elbow plot for KMeans clustering to determine the optimal number of clusters.
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            k_range (tuple): Range of k values to test.
        """
        sse = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df)
            sse.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), sse, marker='o')
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE (Inertia)")
        plt.title("Elbow Method for Optimal k")
        plt.grid(True)
        plt.show()

        return None
    
    def kmeans_clustering_score_plots(self, df, k_range=range(2, 12)):
        """
        Generate KMeans clustering score plots for a range of cluster numbers.

        Parameters:
            df (pd.DataFrame or np.ndarray): The input data for clustering.
            k_range (range): Range of k values to test.

        Returns:
            None
        """
        results = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(df)
            
            # Evaluate scores
            silhouette = silhouette_score(df, labels)
            dbi = davies_bouldin_score(df, labels)

            results.append({
                "k": k,
                "silhouette_score": silhouette,
                "davies_bouldin": dbi
            })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Plotting
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(results_df["k"], results_df["silhouette_score"], marker='o')
        plt.title("Silhouette Score vs K")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(results_df["k"], results_df["davies_bouldin"], marker='o', color='orange')
        plt.title("Davies-Bouldin Score vs K")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Davies-Bouldin Score")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return None
    
    def kmeans_clustering(self, df, k):
        """
        Perform KMeans clustering on the given DataFrame.

        Parameters:
            df (pd.DataFrame or np.ndarray): The input data for clustering.
            k (int): The number of clusters.

        Returns:
            np.ndarray: Cluster labels for each data point.
        """
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters_kmeans = kmeans.fit_predict(df)

        # Evaluate clustering quality
        sil_score = silhouette_score(df, clusters_kmeans)
        self.logger.info(f"Silhouette Score: {sil_score:.3f}")

        dbi = davies_bouldin_score(df, clusters_kmeans)
        self.logger.info(f"Davies-Bouldin Score: {dbi:.3f}")

        return clusters_kmeans
    
    def plot_clusters(self, df, clusters, principal_component_x=0, principal_component_y=1, principal_component_z=None):
        """
        Plots clusters using 2D or 3D PCA-transformed data.

        Parameters:
        - df: ndarray or DataFrame with PCA components
        - clusters: cluster labels
        - principal_component_x, principal_component_y: indexes of PCA components for 2D plot
        - principal_component_z: index of third component for 3D plot (optional)
        """
        
        if principal_component_z is not None and df.shape[1] > principal_component_z:
            # 3D Plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                df[:, principal_component_x], 
                df[:, principal_component_y], 
                df[:, principal_component_z], 
                c=clusters, cmap="plasma", edgecolor='k'
            )
            ax.set_xlabel(f"Principal Component {principal_component_x + 1}")
            ax.set_ylabel(f"Principal Component {principal_component_y + 1}")
            ax.set_zlabel(f"Principal Component {principal_component_z + 1}")
            ax.set_title("Cluster Visualization (PCA 3D)")

            # Create legend
            unique_clusters = np.unique(clusters)
            handles = [
                mpatches.Patch(color=scatter.cmap(scatter.norm(cl)), label=f"Cluster {cl}")
                for cl in unique_clusters
            ]
            ax.legend(handles=handles, title="Cluster Label", loc="upper left")
            plt.show()

        elif df.shape[1] >= 2:
            # 2D Plot
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                df[:, principal_component_x], 
                df[:, principal_component_y], 
                c=clusters, cmap="plasma", edgecolor='k'
            )
            plt.xlabel(f"Principal Component {principal_component_x + 1}")
            plt.ylabel(f"Principal Component {principal_component_y + 1}")
            plt.title("Cluster Visualization (PCA 2D)")
            plt.grid(True)

            unique_clusters = np.unique(clusters)
            handles = [
                mpatches.Patch(color=scatter.cmap(scatter.norm(cl)), label=f"Cluster {cl}")
                for cl in unique_clusters
            ]
            plt.legend(handles=handles, title="Cluster Label", loc="best")
            plt.show()
        
        else:
            self.logger.warning("Not enough components for 2D or 3D plot.")
            
        return None
    
    def dbscan_elbow_plot(self, df, n_neighbors=5):
        """
        Generate a K-distance plot to help determine the optimal epsilon value for DBSCAN.

        Parameters:
            df (pd.DataFrame or np.ndarray): The input data for clustering.
            n_neighbors (int): Number of neighbors to consider for the K-distance.

        Returns:
            None
        """
        # Fit nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors_fit = neighbors.fit(df)
        distances, indices = neighbors_fit.kneighbors(df)

        # Sort distances to find the "knee" point
        distances = np.sort(distances[:, n_neighbors - 1])  # n_neighbors-th nearest neighbor distance
        plt.figure(figsize=(8, 4))
        plt.plot(distances)
        plt.title(f"K-distance plot ({n_neighbors}-th neighbor)")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {n_neighbors}-th nearest neighbor")
        plt.grid(True)
        plt.show()

        return None
    
    def dbscan_clustering_score_plots(self, df, eps_range=np.arange(0.5, 15.0, 0.1), min_samples=5):
        """
        Generate DBSCAN clustering score plots for a range of epsilon values.

        Parameters:
            df (pd.DataFrame or np.ndarray): The input data for clustering.
            eps_range (np.ndarray): Range of epsilon values to test.
            min_samples (int): Minimum number of samples for DBSCAN.

        Returns:
            None
        """
        results = []

        for eps in eps_range:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(df)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # Mask for non-noise points
            mask = labels != -1

            # Only evaluate if we have at least 2 clusters and some valid points
            if n_clusters >= 2 and np.sum(mask) > 10:
                silhouette = silhouette_score(df[mask], labels[mask])
                dbi = davies_bouldin_score(df[mask], labels[mask])
            else:
                silhouette = np.nan
                dbi = np.nan

            results.append({
                "eps": eps,
                "clusters": n_clusters,
                "noise_points": n_noise,
                "silhouette_score": silhouette,
                "davies_bouldin": dbi
            })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Plotting
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(results_df["eps"], results_df["silhouette_score"], marker='o')
        plt.title("Silhouette Score vs Eps")
        plt.xlabel("eps")
        plt.ylabel("Silhouette Score")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(results_df["eps"], results_df["davies_bouldin"], marker='o', color='orange')
        plt.title("Davies-Bouldin Score vs Eps")
        plt.xlabel("eps")
        plt.ylabel("Davies-Bouldin Score")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return None
    
    def dbscan_clustering(self, df, eps, min_samples=5):
        """
        Perform DBSCAN clustering on the given DataFrame.

        Parameters:
            df (pd.DataFrame or np.ndarray): The input data for clustering.
            eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

        Returns:
            np.ndarray: Cluster labels for each data point.
        """
        self.logger.info(f"Running DBSCAN with eps={eps}, min_samples={min_samples}.")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(df)

        # Log the number of clusters and noise points
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        self.logger.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points.")

        return labels

class StatsUtils:

    @staticmethod
    def compute_confidence_intervals(df, cluster_id_col, conf=0.95):
        """
        Compute the confidence interval for each numeric variable per cluster.
        
        Parameters:
        df (DataFrame): Input data frame.
        cluster_id_col (str): Column name for the cluster id (e.g., 'kmeans_cluster_id').
        conf (float): Confidence level (default 0.95).
        
        Returns:
        DataFrame: A summary table with one row per cluster and variable containing:
                    - cluster id
                    - variable name
                    - mean
                    - lower_ci: lower bound of the confidence interval
                    - upper_ci: upper bound of the confidence interval
                    - n: number of observations used
        """
        # Drop 'run_id' if present.
        if 'run_id' in df.columns:
            df = df.drop(columns=['run_id'])
        
        # Drop any other cluster id columns that are not the chosen one.
        cluster_cols = [col for col in df.columns if col.endswith('_cluster_id')]
        for col in cluster_cols:
            if col != cluster_id_col:
                df = df.drop(columns=[col])
        
        # Consider only numeric columns (except the cluster id column).
        numeric_columns = [col for col in df.columns 
                        if col != cluster_id_col and np.issubdtype(df[col].dtype, np.number)]
        
        results = []
        # Group by the cluster id and compute CI for each numeric column.
        for cluster, group in df.groupby(cluster_id_col):
            for col in numeric_columns:
                data = group[col].dropna()
                n = len(data)
                if n > 1:
                    mean_val = data.mean()
                    sem = stats.sem(data)  # standard error of the mean
                    t_val = stats.t.ppf((1 + conf) / 2., n - 1)  # t critical value for n-1 degrees of freedom
                    margin = sem * t_val
                    lower = mean_val - margin
                    upper = mean_val + margin
                else:
                    mean_val = data.mean()
                    lower, upper = np.nan, np.nan  # Not enough data to compute CI
                results.append({
                    cluster_id_col: cluster,
                    'variable': col,
                    'mean': mean_val,
                    'lower_ci': lower,
                    'upper_ci': upper,
                    'n': n
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def plot_distribution_by_cluster(df, cluster_id_col, density=True):
        """
        Plot the normalized distribution of each column (except the cluster id column)
        by cluster in one figure using subplots. Normalization ensures that each histogram
        is scaled so the total area equals 1, allowing for comparison across clusters.
        
        Parameters:
        df (DataFrame): The input DataFrame.
        cluster_id_col (str): The name of the cluster id column to group by (e.g., 'kmeans_cluster_id').
        """
        # Drop 'run_id' if it exists
        if 'run_id' in df.columns:
            df = df.drop(columns=['run_id'])
        
        # Drop any other cluster id columns that are not the chosen one
        cluster_cols = [col for col in df.columns if col.endswith('_cluster_id')]
        for col in cluster_cols:
            if col != cluster_id_col:
                df = df.drop(columns=[col])
        
        # Identify the columns to plot (all columns except the chosen cluster id)
        value_columns = [col for col in df.columns if col != cluster_id_col]
        clusters = df[cluster_id_col].unique()
        
        n_plots = len(value_columns)
        # Compute grid size (roughly square)
        ncols = math.ceil(math.sqrt(n_plots))
        nrows = math.ceil(n_plots / ncols)
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*4))
        
        # Flatten axes array for easy iteration
        if n_plots > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, col in enumerate(value_columns):
            ax = axes[i]
            # Plot normalized histograms for each cluster on the same subplot
            for cluster in clusters:
                data = df[df[cluster_id_col] == cluster][col]
                ax.hist(data, bins=20, alpha=0.5, density=density, label=f'Cluster {cluster}')
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.legend()
        
        # Remove any empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        fig.suptitle(f'Normalized Distributions by {cluster_id_col}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    
    @staticmethod
    def plot_boxplot_with_error_bar(df, ci_df, var_to_plot, cluster_id_col, conf=0.95):
        """
        Plot a boxplot of a variable by cluster and overlay the means with error bars for the confidence intervals.
        parameters:
            df (DataFrame): The input DataFrame containing the data.
            ci_df (DataFrame): DataFrame containing the confidence intervals for each cluster.
            var_to_plot (str): The variable to plot.
            cluster_id_col (str): The name of the cluster id column (e.g., 'kmeans_cluster_id').
            conf (float): Confidence level for the error bars (default is 0.95).
        returns:
            None: Displays the plot.
        """
    
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=cluster_id_col, y=var_to_plot, data=df)

        # Overlay the means with error bars for the confidence intervals.
        cluster_order = sorted(df[cluster_id_col].unique())
        ci_subset = ci_df[ci_df['variable'] == var_to_plot].set_index(cluster_id_col)

        means = [ci_subset.loc[c, 'mean'] for c in cluster_order]
        lower_err = [ci_subset.loc[c, 'mean'] - ci_subset.loc[c, 'lower_ci'] for c in cluster_order]
        upper_err = [ci_subset.loc[c, 'upper_ci'] - ci_subset.loc[c, 'mean'] for c in cluster_order]
        errors = np.array([lower_err, upper_err])

        plt.errorbar(cluster_order, means, yerr=errors, fmt='o', color='red', capsize=5)
        plt.title(f'Boxplot and {conf*100:.0f}% CI for {var_to_plot}')
        plt.show()

    @staticmethod
    def plot_single_group_boxplots_with_error_bars(df, ci_df, variables, cluster_id_col, group_name="", conf=0.95):
        """
        Plot a single group of variables with boxplots by cluster, overlaying means with confidence intervals.
        
        Parameters:
            df (DataFrame): The input DataFrame containing your data.
            ci_df (DataFrame): DataFrame containing the confidence intervals for each variable per cluster.
                            It should have columns: cluster_id_col, 'variable', 'mean', 'lower_ci', 'upper_ci', 'n'
            variables (list): List of variable names (strings) to plot.
            cluster_id_col (str): The name of the cluster id column (e.g., 'kmeans_cluster_id').
            group_name (str): Optional name for the group (e.g., "Resources") for labeling purposes.
            conf (float): Confidence level for the error bars (default is 0.95).
            
        Returns:
            None: Displays a single figure with subplots.
        """
        # Drop 'run_id' if it exists.
        if 'run_id' in df.columns:
            df = df.drop(columns=['run_id'])
        
        # Drop any other cluster id columns that are not the chosen one.
        cluster_cols = [col for col in df.columns if col.endswith('_cluster_id')]
        for col in cluster_cols:
            if col != cluster_id_col:
                df = df.drop(columns=[col])
        
        # Set up the grid for subplots.
        n_vars = len(variables)
        ncols = 2
        nrows = math.ceil(n_vars / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes = axes.flatten()

        # Define the order of clusters (assuming they are sortable).
        cluster_order = sorted(df[cluster_id_col].unique())
        
        # Plot each variable in its own subplot.
        for i, var in enumerate(variables):
            ax = axes[i]
            # Create the boxplot.
            sns.boxplot(x=cluster_id_col, y=var, data=df, ax=ax)
            
            # Extract the confidence interval info for the current variable.
            ci_subset = ci_df[ci_df['variable'] == var].set_index(cluster_id_col)
            means = [ci_subset.loc[c, 'mean'] for c in cluster_order]
            lower_err = [ci_subset.loc[c, 'mean'] - ci_subset.loc[c, 'lower_ci'] for c in cluster_order]
            upper_err = [ci_subset.loc[c, 'upper_ci'] - ci_subset.loc[c, 'mean'] for c in cluster_order]
            errors = np.array([lower_err, upper_err])
            
            # Overlay error bars for the mean and confidence interval.
            ax.errorbar(cluster_order, means, yerr=errors, fmt='o', color='red', capsize=5)
            ax.set_title(var)
        
        # Add a common title to the figure.
        fig.suptitle(f'Boxplots with {conf*100:.0f}% CIs for Group: {group_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()