import pandas as pd
import numpy as np

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
    
    def extract_ts_features(self, group_df, collapse_thresh=0.2, t0=0, t1=5):
        """
        Extracts time series features from a given DataFrame for specified state variables.

        Parameters:
        -----------
        group_df : pd.DataFrame
            DataFrame containing the time series data for the group.
        collapse_thresh : float, optional
            Threshold value to determine the time to collapse (default is 0.2).
        t0 : int, optional
            Initial time point for growth rate calculation (default is 0).
        t1 : int, optional
            Final time point for growth rate calculation (default is 5).

        Returns:
        --------
        pd.Series
            A pandas Series containing the extracted features for each state variable.
            The features include:
            - final value
            - maximum value
            - minimum value
            - area under the curve (AUC)
            - time to collapse
            - standard deviation
            - growth rate

        State Variables:
        ----------------
        - Resources
        - Economy
        - Bureaucracy
        - Pollution
        """
        features = {}
        state_vars = ["Resources", "Economy", "Bureaucracy", "Pollution"]

        for state_var in state_vars:
            features[state_var + "_final"] = self.get_final_value(group_df, state_var)
            features[state_var + "_max"] = self.get_max_value(group_df, state_var)
            features[state_var + "_min"] = self.get_min_value(group_df, state_var)
            features[state_var + "_auc"] = self.get_auc(group_df, state_var)
            features[state_var + "_time_to_collapse"] = self.get_time_to_collapse(group_df, state_var, collapse_thresh)
            features[state_var + "_std"] = self.get_std(group_df, state_var)
            features[state_var + "_growth_rate"] = self.get_growth_rate(group_df, state_var, t0, t1)

        return pd.Series(features)