import numpy as np
from pymoo.core.problem import Problem
import torch

class CustomProblem(Problem):
    def __init__(self, data_interface, model_interface, query_instance, observed_instances, target_cf_class, feature_weights_list, features_to_vary, submethod, total_CFs):
        num_features = len(data_interface.encoded_feature_names)
        feature_min, feature_max = data_interface.get_min_max(normalized=True)

        feature_min, feature_max = np.array(feature_min), np.array(feature_max)

        super().__init__(n_var=num_features, n_obj=5, n_constr=0, xl=feature_min, xu=feature_max)

        self.data_interface = data_interface
        self.query_instance = query_instance
        self.model_interface = model_interface
        self.observed_instances = observed_instances
        self.target_cf_class = target_cf_class
        self.feature_weights_list = feature_weights_list
        self.features_to_vary = features_to_vary
        self.submethod = submethod
        self.total_CFs = total_CFs

        if self.features_to_vary == "all":
            self.features_to_vary = self.data_interface.encoded_feature_names
        
        for idx, feature_name in enumerate(data_interface.encoded_feature_names):
            if feature_name not in self.features_to_vary:
                self.xl[idx] = query_instance[idx]
                self.xu[idx] = query_instance[idx]

    def _evaluate(self, x, out, *args, **kwargs):
        yloss = np.array([self.compute_yloss(cf_instance) for cf_instance in x])
        proximity = np.array([self.compute_proximity_loss(self.query_instance, cf_instance) for cf_instance in x])
        sparsity = np.array([self.compute_sparsity_loss(self.query_instance, cf_instance) for cf_instance in x])
        plausibility = np.array([self.compute_plausibility_loss(cf_instance, self.observed_instances) for cf_instance in x])
        regularization = np.array([self.compute_regularization_loss(cf_instance) for cf_instance in x])

        out["F"] = np.column_stack([yloss, proximity, sparsity, plausibility, regularization])

    def compute_yloss(self, cf_instance):
        # Implement the computation of the yloss objective for a given CF instance
        cf_instance = torch.Tensor(cf_instance)
        predictions = self.model_interface.predict(cf_instance)
        if not isinstance(predictions, np.ndarray):
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()  # Convert from PyTorch tensor to numpy array
            else:
                predictions = np.array(predictions)
        logits = np.log(np.abs(predictions - 1e-6) / np.abs(1 - predictions - 1e-6))
        all_ones = np.ones_like(self.target_cf_class)
        labels = 2 * self.target_cf_class - all_ones
        if not isinstance(labels, np.ndarray):
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()  # Convert from PyTorch tensor to numpy array
            else:
                labels = np.array(labels)
        temp_loss = all_ones - labels * logits
        yloss = np.maximum(0, temp_loss)  # Applying ReLU
        return np.mean(yloss)

    def compute_proximity_loss(self, query_instance, cf_instance):
        query_instance = query_instance.detach().numpy()
        return np.mean(np.abs(cf_instance - query_instance) * self.feature_weights_list)
    
    def compute_dist(self, x1, x2):
        return np.sum(np.abs(x1 - x2) * self.feature_weights_list, axis=0)

    def dpp_style(self, cf_instance, submethod):
        det_entries = np.ones((self.total_CFs, self.total_CFs))
        for i in range(self.total_CFs):
            for j in range(self.total_CFs):
                det_entries[i, j] = self.compute_dist(cf_instance[i], cf_instance[j])
        if submethod == "inverse_dist":
            det_entries = 1.0 / (1.0 + det_entries)
        if submethod == "exponential_dist":
            det_entries = 1.0 / np.exp(det_entries)

        det_entries += np.eye(self.total_CFs) * 0.0001
        return np.linalg.det(det_entries)

    def compute_diversity_loss(self, cf_instance, submethod):
        if self.total_CFs == 1:
            return np.array(0.0)
        return self.dpp_style(cf_instance, submethod)
        
    def compute_sparsity_loss(self, query_instance, cf_instance):
        query_instance = query_instance.detach().numpy()
        return np.count_nonzero(cf_instance - query_instance) / len(self.data_interface.encoded_feature_names)

    def compute_plausibility_loss(self, cf_instance, observed_instances, k=5):
        cf_instance = np.expand_dims(cf_instance, axis=1)
        observed_instances = np.array(observed_instances.values, dtype=np.float32)
        diff = np.abs(cf_instance[:, None, :] - observed_instances[None, :, :])
        distances = np.sum(diff, axis=2)
        sorted_distances = np.sort(distances, axis=1)
        nearest_distances = sorted_distances[:, :k]
        return np.mean(nearest_distances)

    def compute_regularization_loss(self, cf_instance):
        cf_instance = np.expand_dims(cf_instance, axis=1)
        regularization_loss = 0
        for v in self.data_interface.encoded_cat_feature_indices:
            regularization_loss += np.sum(np.square(np.sum(cf_instance[:, v[0]:v[-1]+1], axis=1) - 1.0))
        return regularization_loss