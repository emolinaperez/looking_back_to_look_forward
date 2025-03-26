import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def extract_ts_features(self, group_df, collapse_thresh=0.2):
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
            
            features[state_var + "_max_min_diff"] = features[state_var + "_max"] - features[state_var + "_min"]
            features[state_var + "_final_min_diff"] = features[state_var + "_final"] - features[state_var + "_min"]
            features[state_var + "_final_initial_diff"] = features[state_var + "_final"] - group_df[state_var].iloc[0]

            features[state_var + "_time_to_collapse"] = self.get_time_to_collapse(group_df, state_var, collapse_thresh)
            features[state_var + "_std"] = self.get_std(group_df, state_var)
            features[state_var + "_growth_rate_t4"] = self.get_growth_rate(group_df, state_var, 0, 4)
            features[state_var + "_growth_rate_t25"] = self.get_growth_rate(group_df, state_var, 0, 25)
            
        
            # Early delta: difference between the 5th period and the initial value
            early_delta = group_df[state_var].iloc[5] - group_df[state_var].iloc[0]
            features[state_var + "_early_delta"] = early_delta
            # Mid delta: difference between the 15th period and the initial value
            mid_delta = group_df[state_var].iloc[20] - group_df[state_var].iloc[0]
            features[state_var + "_mid_delta"] = mid_delta
            
            # Early average difference: average change between consecutive values over the first 5 periods
            early_avg_diff = group_df[state_var].iloc[:5].diff().mean()
            features[state_var + "_early_avg_diff"] = early_avg_diff

            # Mid average difference: average change between consecutive values over the first 20 periods
            mid_avg_diff = group_df[state_var].iloc[:20].diff().mean()
            features[state_var + "_mid_avg_diff"] = mid_avg_diff

        return pd.Series(features)

# Example usage:
# Assuming output_df is the DataFrame containing the simulation results with columns:
#   Resources, Economy, Bureaucracy, Pollution, time, run_id
#
# feature_extractor = FeatureExtractor()
# features_df = output_df.groupby("run_id").apply(feature_extractor.extract_ts_features).reset_index()

class EDAUtils:

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

    def plot_corr_heatmap(df, figsize=(20, 15)):
        # Correlation heatmap to inspect relationships between features
        plt.figure(figsize=figsize)
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.show()

