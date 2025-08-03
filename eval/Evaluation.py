import torch

class Evaluate:

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: DataInterface object.
        :param model_interface: ModelInterface object.
        """
        self.data_interface = data_interface
        self.model_interface = model_interface

    def evaluate_proximity(self, query_instance, cf_instances):
        """Evaluates the proximity of the counterfactuals to the query instance.

        :param query_instance: PyTorch tensor of the query instance.
        :param cf_instances: PyTorch tensor of the counterfactual instances.
        
        :return: Proximity score.
        """
        proximity = torch.mean(torch.abs(cf_instances - query_instance))
        return proximity
    
    def evaluate_sparsity(self, query_instance, cf_instances, epsilon=torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps))):
        """
        Evaluates the sparsity, or how many features are changed in the cf instances compared with the query instance.
        
        :param query_instance: PyTorch tensor of the original instance.
        :param cf_instances: PyTorch tensor of counterfactual instances (2D tensor; each row is a counterfactual).
        :param epsilon: Threshold below which differences in numerical features are ignored.
        
        :return: Sparsity score.
        """    
        # Initialize a boolean tensor for differences
        differences = torch.zeros_like(cf_instances, dtype=torch.bool)
        
        # For numerical features, check if differences are greater than epsilon
        differences = torch.abs(cf_instances - query_instance[0]) > epsilon
        
        # Count the number of true differences for each counterfactual
        differences = torch.sum(differences, axis=1)
        sparsity = torch.mean(differences.float())
        sparsity = sparsity / len(query_instance[0])
        return sparsity

    def compute_dpp(self, cf_instances):
        """
        Computes the DPP of the matrix of counterfactual instances.

        :param cf_instances: PyTorch tensor of counterfactual instances.

        :return: DPP score.
        """
        total_CFs = cf_instances.size(0)
        det_entries = torch.ones(total_CFs, total_CFs)
        for i in range(total_CFs):
            for j in range(total_CFs):
                det_entries[i, j] = torch.sum((torch.abs(cf_instances[i] - cf_instances[j])), dim = 0)

        det_entries = 1.0 / (1.0 + det_entries)

        det_entries += torch.eye(total_CFs) * 0.0001
        return torch.det(det_entries)

    def evaluate_diversity(self, cf_instances):
        """
        Computes the diversity of the counterfactual instances.

        :param cf_instances: PyTorch tensor of counterfactual instances.

        :return: Diversity score.
        """
        total_CFs = cf_instances.size(0)
        if total_CFs == 1:
            return torch.tensor(0.0)
        else:
            return self.compute_dpp(cf_instances)
    
    def compute_distances(self, cf_instances, observed_instances):
        """
        Calculate the distances between each counterfactual instance and each observed instance.

        :param cf_instances: PyTorch tensor of counterfactual instances.
        :param observed_instances: PyTorch tensor of observed instances.

        :return: Distances between each counterfactual instance and each observed instance.
        """
        n_cf = cf_instances.size(0)
        observed_instances = torch.tensor(observed_instances.to_numpy(), dtype=torch.float)
        n_observed = observed_instances.size(0)

        cf_expanded = cf_instances.unsqueeze(1).expand(n_cf, n_observed, -1)
        observed_expanded = observed_instances.unsqueeze(0).expand(n_cf, n_observed, -1)

        distances = torch.norm(cf_expanded - observed_expanded, p=1, dim=2)
        return distances

    def evaluate_plausibility(self, cf_instances, observed_instances, k):
        """
        Evaluates the plausibility of the counterfactuals.

        :param cf_instances: PyTorch tensor of counterfactual instances.
        :param observed_instances: PyTorch tensor of observed instances.
        :param k: Number of nearest neighbors to consider.

        :return: Plausibility score.
        """
        distances = self.compute_distances(cf_instances, observed_instances)
        _, indices = torch.topk(distances, k, largest=False)
        
        k_nearest_distances = torch.gather(distances, 1, indices)
        plausibility_loss = k_nearest_distances.mean()
        return plausibility_loss
    
    def evaluate_confidence(self, cf_instances, desired_class):
        """
        Evaluates the confidence of the counterfactuals.

        :param cf_instances: PyTorch tensor of counterfactual instances.
        :param desired_class: Desired class for the counterfactual instances.
        
        :return: Confidence scores.
        """
        confidence_scores = self.model_interface.predict(cf_instances)
        confidence_scores = torch.mean(confidence_scores)
        if desired_class == 1:
            return confidence_scores
        elif desired_class == 0:
            return 1 - confidence_scores
        
    def evaluate_confidence_multiclass(self, cf_instances, desired_class, num_classes):
        confidence_scores = self.model_interface.predict(cf_instances)
        if isinstance(desired_class, int):
            desired_class = torch.tensor([desired_class] * cf_instances.size(0)).unsqueeze(1)
        confidence_scores = torch.gather(confidence_scores, 1, desired_class)
        mean_confidence = torch.mean(confidence_scores)
        print('Mean confidence:', mean_confidence)
        mean_confidence = mean_confidence / num_classes
        return mean_confidence
    
    def evaluate(self, query_instance, cf_instances, observed_data, desired_class, num_classes, k):
        """
        Evaluates the counterfactuals.

        :param query_instance: PyTorch tensor of the query instance.
        :param cf_instances: PyTorch tensor of counterfactual instances.
        :param observed_data: PyTorch tensor of the observed data.
        :param k: Number of nearest neighbors.
        
        :return: Evaluation scores.
        """
        proximity_scores = self.evaluate_proximity(query_instance, cf_instances)
        sparsity_scores = self.evaluate_sparsity(query_instance, cf_instances)
        diversity_scores = self.evaluate_diversity(cf_instances)
        plausibility_scores = self.evaluate_plausibility(cf_instances, observed_data, k)
        confidence_scores = self.evaluate_confidence_multiclass(cf_instances, desired_class, num_classes)

        print('Proximity:', proximity_scores)
        print('Sparsity:', sparsity_scores)
        print('Diversity:', diversity_scores)
        print('Plausibility:', plausibility_scores)
        print('Confidence:', confidence_scores) 