import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import defaultdict
import torch

class DataInterface:
    def __init__(self, **params):

        """Init method

        :param dataframe: Pandas DataFrame.
        :param target: Outcome feature name.
        :param continuous_features: List of continuous feature names. If 'all', all features except the target are considered continuous.
        """

        if isinstance(params['dataframe'], pd.DataFrame):
            self.df = params['dataframe']
        else:
            raise ValueError("Data should be a Pandas DataFrame.")

        if type(params['target']) is str:
            self.target = params['target']
        else:
            raise ValueError("Target feature name not provided.")

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_names = self.df.columns[0:-1].to_list()
        self.train_df, self.test_df = self.tt_split(self.df)
        self.features = self.train_df.iloc[:, 0:-1].values
        self.targets = self.train_df.iloc[:, -1:].values
        self.labels = self.df.iloc[:, -1:].values

        self.norm_features = self.feature_scaler.fit_transform(self.train_df.iloc[:, 0:-1].values)        
        self.norm_targets = self.target_scaler.fit_transform(self.train_df.iloc[:, -1:])

        if params['continuous_features'] == 'all':
            self.cont_feature_names = self.df.columns.tolist()
            self.cont_feature_names.remove(self.target)
        else:
            if type(params['continuous_features']) is list:
                self.cont_feature_names = params['continuous_features']
            else:
                raise ValueError("Continuous features not provided.")

        self.cat_feature_names = [name for name in self.df.columns.tolist()
                if name not in self.cont_feature_names + [self.target]]

        if len(self.cat_feature_names) > 0:
            self.encoded_features = pd.get_dummies(data = self.df, columns = self.cat_feature_names)
            self.encoded_feature_names  = self.encoded_features.columns.tolist()
            self.encoded_feature_names.remove(self.target)
        else:
            self.encoded_features = self.df
            self.encoded_feature_names = self.feature_names

        self.encoded_features = self.encoded_features.drop(columns=[self.target])
        for column in self.encoded_features.columns:
            if self.encoded_features[column].dtype == 'bool':
                self.encoded_features[column] = self.encoded_features[column].astype(int)

        self.cont_features = self.df[self.cont_feature_names]
        self.cat_features = self.df[self.cat_feature_names]
        self.norm_cont_features = self.feature_scaler.fit_transform(self.cont_features)
        self.norm_features = np.concatenate((self.norm_cont_features, self.cat_features), axis=1)    
        self.norm_encoded_features = self.encoded_features.copy()
        for i, column in enumerate(self.cont_feature_names):
            if column in self.norm_encoded_features.columns:
                self.norm_encoded_features[column] = self.norm_cont_features[:, i]
        
        self.cont_feature_precisions = self.get_feature_precisions(self.cont_features)
        
        self.encoded_cat_feature_indices = self.get_encoded_cat_feature_indices()

    def check_feature_ranges(self):
        """
        Sets the permitted range for each feature based on the training data.
        """
        for feature in self.feature_names:
            self.permitted_range[feature] = [self.train_df[feature].min(), self.train_df[feature].max()]
        return True

    def norm_data(self, df):
        """
        Normalizes continuous features to make them fall in the range [0,1].
        """
        return self.feature_scaler.transform(df)

    def denorm_data(self, df):
        """
        De-normalizes continuous features from [0,1] range to original range.
        """
        continuous_indices = self.get_cont_feature_indices()
        continuous_data = df[:, continuous_indices]
        continuous_data_denorm = self.feature_scaler.inverse_transform(continuous_data)
        df[:, continuous_indices] = continuous_data_denorm

        return df

    def get_cont_feature_indices(self):
        """
        Returns indices of continuous features in the dataset.
        """
        continuous_feature_names = self.cont_feature_names
        indices = [self.encoded_feature_names.index(name) for name in continuous_feature_names]

        return indices

    def get_cat_mapping(self, df):
        """
        Maps one-hot encoded features back to original feature names.
        """
        mapping = defaultdict(list)
        for column in df.columns:
            parts = column.split('_')
            if len(parts) > 1:
                original_feature_name = '_'.join(parts[:-1])
                mapping[original_feature_name].append(column)
        return dict(mapping)
    
    def onehot_to_cat(self, df, cat_mapping):
        """
        Converts one-hot encoded features back to original categorical features.
        """
        for original_feature, onehot_features in cat_mapping.items():
            df[original_feature] = df[onehot_features].idxmax(axis=1)
            df[original_feature] = df[original_feature].apply(lambda x: int(x.split('_')[-1]))
            df = df.drop(columns=onehot_features)
        return df
    
    def get_feature_precisions(self, df):
        """
        Captures the precision (number of decimal places) for each feature in the dataframe.
        """
        precisions = {}
        for col in df.columns:
            sample_value = df[col].dropna().iloc[0]
            if isinstance(sample_value, float):
                decimal_places = str(sample_value)[::-1].find('.')
                precisions[col] = decimal_places
            else:
                precisions[col] = 0
        return precisions
    
    def apply_feature_precisions(self, df, precision_dict):
        """
        Applies the precision to the dataframe columns based on the precision dictionary.
        """
        for col, precision in precision_dict.items():
            df[col] = df[col].round(precision)
        return df

    def norm_target(self, target):
        """
        Normalizes the target feature.
        """
        return self.target_scaler.transform(target)

    def denorm_target(self, target):
        """
        De-normalizes the target feature.
        """
        return self.target_scaler.inverse_transform(target)

    def tt_split(self, data):
        """
        Splits the data into training and testing sets.
        """
        train_df, test_df = train_test_split(
            data, test_size = 0.2, shuffle = False)
        return train_df, test_df

    def get_mads(self, normalized = True, feature_indices = None):
        """
        Computes Median Absolute Deviation of features.
        """
        data_to_use = self.norm_encoded_features if normalized else self.encoded_features
        if feature_indices is not None:
            data_to_use = data_to_use.iloc[:, feature_indices]
        median_of_data = np.median(data_to_use, axis=0)
        absolute_deviation = np.abs(data_to_use - median_of_data)
        absolute_deviation = np.where(absolute_deviation == 0, 1e-8, absolute_deviation)
        mads = np.median(absolute_deviation, axis=0)
        
        return mads

    def get_stds(self, normalized = True):
        """
        Computes the standard deviation of features.
        """
        if not normalized:
            self.stds = np.std(self.encoded_features)
        else:
            self.stds = np.std(self.norm_features)
        return self.stds

    def get_min_max(self, normalized = True):
        """
        Computes max/min of features.
        """
        if not normalized:
            self.minx, self.maxx =  self.encoded_features.min(0), self.encoded_features.max(0)
        else:
            self.minx, self.maxx =  self.norm_encoded_features.min(0), self.norm_encoded_features.max(0)
        return self.minx, self.maxx

    def get_features_to_vary_mask(self, features_to_vary):
        """
        Creates a mask to vary specified features.
        """
        if features_to_vary == "all":
            return np.ones(len(self.encoded_feature_names))
        mask = np.zeros(len(self.encoded_feature_names))
        for feature in features_to_vary:
            if feature in self.cat_feature_names:
                for i, encoded_feature in enumerate(self.encoded_feature_names):
                    if encoded_feature.startswith(feature + "_"):
                        mask[i] = 1
            else:
                if feature in self.encoded_feature_names:
                    index = self.encoded_feature_names.index(feature)
                    mask[index] = 1
        return mask
    
    def get_immutable_features_mask(self, immutable_features):
        """
        Creates a mask to fix specified immutable features.
        """
        mask = np.ones(len(self.encoded_feature_names))  # Start with all features set to 1 (varyable)

        for feature in immutable_features:
            if feature in self.cat_feature_names:
                # Handle categorical features
                for i, encoded_feature in enumerate(self.encoded_feature_names):
                    if encoded_feature.startswith(feature + "_"):
                        mask[i] = 0  # Set to 0 to fix this categorical feature
            else:
                # Handle non-categorical features
                if feature in self.encoded_feature_names:
                    index = self.encoded_feature_names.index(feature)
                    mask[index] = 0  # Set to 0 to fix this feature

        return mask

    def get_features_to_vary_weights(self, feature_weights):
        """
        Gets predefined weights for features to vary.
        """
        weights = np.zeros(len(self.encoded_feature_names)) 
        for key, value in feature_weights.items():
            idx = self.encoded_feature_names.index(key)
            weights[idx] = float(value)

        return weights
    
    def get_encoded_cat_feature_indices(self):
        """
        Returns the indices of one-hot encoded categorical features.
        """
        cols = []
        for col_parent in self.cat_feature_names:
            temp = [self.encoded_feature_names.index(col) for col in self.encoded_feature_names if col.startswith(col_parent) and col not in self.cont_feature_names]
            cols.append(temp)

        return cols
    
    def get_feature_index(self, feature):
        """
        Returns the index of a feature in the dataset. 
        """
        return self.encoded_feature_names.index(feature)
    
    def onehot_to_original(self, df, cat_feature_indices):
        """
        Reverts one-hot encoded features back to original categorical features.
        """
        original_df = df.copy()
        for feature_indices in cat_feature_indices:
            feature_name = feature_indices.rsplit('_', 1)[0]
            original_df[feature_name] = df[feature_indices].idxmax(axis=1).apply(lambda x: x.split('_')[-1])
            original_df.drop(columns=feature_indices, inplace=True)
        return original_df

    def get_feature_categories(self, feature):
        """
        Returns the categories of a categorical feature.
        """
        base_feature = feature.split('_')[0]
        if base_feature in self.cat_feature_names:
            pattern = base_feature + '_'
            categories = [col.split(pattern)[-1] for col in self.encoded_features.columns if col.startswith(pattern)]
            if categories:
                return categories
            else:
                raise KeyError(f"No categories found for feature {feature}")
        else:
            raise KeyError(f"No categories found for feature {base_feature}")
        
    def get_feature_name(self, index):
        """
        Returns the name of a feature given its index.
        """
        return [self.encoded_feature_names[i] for i in index]
    
    def remove_suffixes(self, feature_names):
        """
        Removes suffixes like _0, _1, etc. from the feature names.
        
        :param feature_names: List of feature names
        :return: List of feature names with suffixes removed
        """
        new_feature_names = []
        for name in feature_names:
            if '_' in name:
                base_name = '_'.join(name.split('_')[:-1])
                new_feature_names.append(base_name)
            else:
                new_feature_names.append(name)
        return new_feature_names
    
    def features_to_vary_except(self, immutable_features):
        features_to_vary_except = []
        for feature in self.feature_names:
            if feature not in immutable_features:
                features_to_vary_except.append(feature)
        return features_to_vary_except
    
    def get_feature_range(self, feature):
        """
        Returns the range of values for a feature.
        """
        return [self.df[feature].min(), self.df[feature].max()]
    
    def get_quantiles(self, quantile = 0.05, normalized = True):
        """
        Computes the quantiles of features.
        """
        quantile = np.zeros(len(self.encoded_feature_names))
        if normalized:
            quantile = [np.quantile(abs(list(set(self.norm_features[i].tolist())) - np.median(list(set(self.norm_features[i].tolist())))), quantile) for i in self.norm_features]
            return quantile

        else:
            quantile = [np.quantile(abs(list(set(self.norm_features[i].tolist())) - np.median(list(set(self.norm_features[i].tolist())))), quantile) for i in self.norm_features]
            return quantile
        
    def prepare_instance(self, query_instance, normalized=False):
        """
        Converts the query_instance to a DataFrame and prepares it for prediction.
        """
        if isinstance(query_instance, list):
            test = pd.DataFrame([query_instance], columns=self.feature_names)
        else:
            raise ValueError("Unsupported data type of query_instance. Please provide a list.")

        cont_features = test[self.cont_feature_names]
        cat_features = test[self.cat_feature_names]

        if normalized:
            cont_features = self.norm_data(cont_features)

        if self.cat_feature_names:
            onehot_encoded_data = pd.get_dummies(cat_features, columns=self.cat_feature_names)
            onehot_encoded_data.columns = onehot_encoded_data.columns.str.replace('\.0', '', regex=True)
            onehot_encoded_data = onehot_encoded_data.astype(int)
            missing_cols = set(self.encoded_feature_names) - set(onehot_encoded_data.columns)
            for col in missing_cols:
                onehot_encoded_data[col] = 0
            onehot_encoded_data = onehot_encoded_data[self.encoded_feature_names]
            cont_features = pd.DataFrame(cont_features, columns=self.cont_feature_names)
            cat_features = onehot_encoded_data.drop(columns=self.cont_feature_names)
            prepared_instance = pd.concat([cont_features, cat_features], axis=1)
        else:
            prepared_instance = cont_features

        return np.array(prepared_instance, dtype=np.float32)
