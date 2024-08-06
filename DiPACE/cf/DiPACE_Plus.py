import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class DiPACE_Plus:

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params. (See data/Data_Interface.py)
        :param model_interface: an interface class to access trained ML model. (See model/Model_Interface.py)

        """
        self.data_interface = data_interface
        self.model_interface = model_interface
        self.minx, self.maxx = self.data_interface.get_min_max(normalized=True)
        self.feature_min = torch.tensor(self.minx.to_numpy(), dtype=torch.float32)
        self.feature_max = torch.tensor(self.maxx.to_numpy(), dtype=torch.float32)

    def generate_counterfactuals(self, query_instance, total_CFs, immutable_features=[], acceptable_ranges={}, acceptable_categories={}, acceptable_directions={}, desired_class="opposite", proximity_weight=0.4, diversity_weight=0.4, sparsity_weight = 0.5, plausibility_weight = 0.4, categorical_penalty=0.1, proximity_threshold=0.4, sparsity_threshold=0.05, diversity_threshold=0.8, plausibility_threshold=1.5, features_to_vary="all", learning_rate=0.005, min_iter=500, max_iter=100000, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, tie_random=False, pred_loss_type='hinge', high_loss_threshold=1.5, stopping_threshold=0.5, penalty_scale = 0.1, perturb_scale = 0.5, k=3): 
        """
        Generates counterfactual explanations with the DiPACE algorithm

        :param query_instance: A list of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.
        
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param sparsity_weight: A positive float. Larger this weight, more sparse the counterfactuals are.
        :param plausibility_weight: A positive float. Larger this weight, more plausible the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.

        :param proximity_threshold: A float. Threshold for proximity loss values. If the loss exceeds this value, the counterfactuals are perturbed.
        :param sparsity_threshold: A float. Threshold for sparsity loss values. If the loss exceeds this value, the counterfactuals are perturbed.
        :param diversity_threshold: A float. Threshold for diversity loss values. If the loss exceeds this value, the counterfactuals are perturbed.
        :param plausibility_threshold: A float. Threshold for plausibility loss values. If the loss exceeds this value, the counterfactuals are perturbed.
        :param high_loss_threshold: A float. Threshold for high loss values. If the loss exceeds this value, the relevant features are perturbed.

        :param penalty_scale: A float. A scale for the penalties.
        :param perturb_scale: A float. A scale for the perturbations.

        :param immutable_features: A list of feature names that should not be changed.
        :param acceptable_ranges: A dictionary of feature names and a list of acceptable values for each feature.
        :param acceptable_categories: A dictionary of feature names and a list of acceptable categories for each feature.
        :param acceptable_directions: A dictionary of feature names and a direction for each feature. Takes "increase", "decrease" or None.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.

        :param pred_loss_type: Metric for pred loss of the optimization function. Takes "cross_entropy" or "hinge".

        :param learning_rate: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence. Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.

        :return: A Counterfactuals object.
        """
        self.total_CFs = total_CFs
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight
        self.plausibility_weight = plausibility_weight
        self.categorical_penalty = categorical_penalty
        self.proximity_threshold = proximity_threshold
        self.sparsity_threshold = sparsity_threshold
        self.diversity_threshold = diversity_threshold
        self.plausibility_threshold = plausibility_threshold
        self.penalty_scale = penalty_scale
        self.perturb_scale = perturb_scale
        self.pred_loss_type = pred_loss_type
        
        query_instance = self.data_interface.prepare_instance(query_instance, normalized = True)
        query_instance = torch.FloatTensor(query_instance)
                
        vary_mask = self.data_interface.get_features_to_vary_mask(features_to_vary)
        vary_mask = torch.LongTensor(vary_mask)

        fix_mask = self.data_interface.get_immutable_features_mask(immutable_features)
        fix_mask = torch.FloatTensor(fix_mask)
        
        names = []
        indices = []
        min_vals = []
        max_vals = []
        for name, values in acceptable_ranges.items():
            # Get the index of the feature
            names.append(name)
            index = self.data_interface.get_feature_index(name)
            indices.append(index)
            
            # Create a sample with only the current feature set to the value
            n_features = len(self.data_interface.cont_feature_names)
            samples = samples = np.zeros((len(values), n_features))
            for i, value in enumerate(values):
                samples[i, index] = value
                
            # Normalize the sample
            norm_sample = self.data_interface.norm_data(samples)
                
            # Extract the min and max values for the normalized feature
            min_val = norm_sample[0][index]
            max_val = norm_sample[1][index]
            min_vals.append(min_val)
            max_vals.append(max_val)

        cf_instances = torch.rand(total_CFs, query_instance.shape[1])
        cf_instances = torch.FloatTensor(cf_instances)

        for name, index, min_val, max_val in zip(names, indices, min_vals, max_vals):
            # Clamp the values in the counterfactual instances to the specified range
            cf_instances[:, index] = torch.clamp(cf_instances[:, index], min=min_val, max=max_val)

        cf_instances = vary_mask * cf_instances + (1 - vary_mask) * query_instance
        cf_instances = fix_mask * cf_instances + (1 - fix_mask) * query_instance

        continuous_feature_indices = [self.data_interface.encoded_feature_names.index(name) for name in self.data_interface.cont_feature_names if name in self.data_interface.encoded_feature_names]

        mads_continuous = self.data_interface.get_mads(normalized=True, feature_indices=continuous_feature_indices)
        inverse_mads_continuous = 1 / np.maximum(mads_continuous, 1e-6)
        inverse_mads_continuous = np.round(inverse_mads_continuous, 2)

        self.feature_weights_list = np.ones(query_instance.shape[1])

        for index, weight in zip(continuous_feature_indices, inverse_mads_continuous):
            self.feature_weights_list[index] = weight

        self.feature_weights_list = torch.from_numpy(self.feature_weights_list).float()

        self.optimizer = torch.optim.Adam([cf_instances], lr = learning_rate)

        self.model_interface.load_model()
        test_pred = torch.round(self.model_interface.predict(query_instance)[0])
        if desired_class == "opposite":
            self.target_cf_class = 1.0 - torch.round(test_pred)
        else:
            self.target_cf_class = desired_class

        print('Desired class:', self.target_cf_class)
        print('Original class', test_pred)

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.loss_diff_thres = loss_diff_thres
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        
        self.stopping_threshold = stopping_threshold

        self.tie_random = tie_random

        self.best_backup_cfs = [0]*total_CFs
        self.best_backup_cfs_preds = [0]*total_CFs
        self.min_dist_from_threshold = 100

        iterations_list = []
        loss_values = []
        pred_losses = []
        proximity_losses = []
        diversity_losses = []
        sparsity_losses = []
        plausibility_losses = []
        perturbation_points = []
        
        global_iterations = 0
        iterations = 0
        loss_diff = 0
        prev_loss = 0
        perturbation_attempts = 0
        max_perturbation_attempts = 5
        best_loss = float('inf')
        best_cf_instances = cf_instances.clone().detach()
        
        while self.stop_loop(iterations, loss_diff, cf_instances) is False:
            with torch.no_grad():
                cf_instances.clamp_(min=self.feature_min, max=self.feature_max)
                if acceptable_ranges:
                    for name, index, min_val, max_val in zip(names, indices, min_vals, max_vals):
                        cf_instances[:, index] = torch.clamp(cf_instances[:, index], min=min_val, max=max_val)
                #Need to fix this:
                if acceptable_categories:
                    for feature, values in acceptable_categories.items():
                        if feature in self.data_interface.cat_feature_names:
                            index = self.data_interface.feature_names.index(feature)
                            for i in range(total_CFs):
                                if cf_instances[i, index] not in values:
                                    cf_instances[i, index] = query_instance[:, index]
                if acceptable_directions:
                    for name, direction in acceptable_directions.items():
                        index = self.data_interface.get_feature_index(name)
                        if direction == 'increase':
                            cf_instances[:, index] = torch.max(cf_instances[:, index], query_instance[:, index])
                        elif direction == 'decrease':
                            cf_instances[:, index] = torch.min(cf_instances[:, index], query_instance[:, index])

            cf_instances.requires_grad = True
            self.optimizer.zero_grad()
            loss_value = self.compute_loss(query_instance, cf_instances, k)
            pred_loss = self.compute_pred_loss(cf_instances)
            proximity_loss = self.compute_proximity_loss(query_instance, cf_instances)
            diversity_loss = self.compute_diversity_loss(cf_instances)
            sparsity_loss = self.compute_sparsity_loss(query_instance, cf_instances)
            plausibility_loss = self.compute_plausibility_loss(cf_instances, self.data_interface.norm_encoded_features, k)
            loss_value.backward(retain_graph = True)
            cf_instances.grad = cf_instances.grad * vary_mask
            cf_instances.grad = cf_instances.grad * fix_mask
            self.optimizer.step()

            iterations_list.append(global_iterations)
            loss_values.append(loss_value.item())
            pred_losses.append(pred_loss.item())
            proximity_losses.append(proximity_loss.item())
            diversity_losses.append(-diversity_loss.item())
            sparsity_losses.append(sparsity_loss.item())
            plausibility_losses.append(plausibility_loss.item())

            if verbose:
                if (iterations) % 50 == 0:
                    print('step %d,  loss=%g' % (iterations+1, loss_value))

            loss_diff = abs(loss_value-prev_loss)
            prev_loss = loss_value
            iterations += 1
            global_iterations += 1

            if self.stop_loop(iterations, loss_diff, cf_instances): 
                final_loss = self.compute_loss(query_instance, cf_instances, k)
                final_pred_loss = self.compute_pred_loss(cf_instances)
                final_proximity_loss = self.compute_proximity_loss(query_instance, cf_instances)
                final_diversity_loss = self.compute_diversity_loss(cf_instances)
                final_sparsity_loss = self.compute_sparsity_loss(query_instance, cf_instances)
                final_plausibility_loss = self.compute_plausibility_loss(cf_instances, self.data_interface.norm_encoded_features, k)

                print('Final Loss:', final_loss)
                print('Final Pred Loss:', final_pred_loss)
                print('Final Proximity Loss:', final_proximity_loss)
                print('Final Diversity Loss:', final_diversity_loss)
                print('Final Sparsity Loss:', final_sparsity_loss)
                print('Final Plausibility Loss:', final_plausibility_loss)

                if final_loss > high_loss_threshold:
                    print("Unacceptably high loss. Perturbing relevant features.")

                    if final_loss < best_loss:
                        best_loss = final_loss
                        best_cf_instances = cf_instances.clone().detach()
                        perturbation_attempts = 0
                    else:
                        perturbation_attempts += 1

                    if perturbation_attempts >= max_perturbation_attempts:
                        print("Maximum perturbation attempts reached. Stopping optimization.")
                        break

                    perturbed_cf_instances = cf_instances.clone().detach()
                    for i in range(perturbed_cf_instances.shape[1]):
                        if abs(cf_instances[0, i] - query_instance[0, i]) > 1e-5:
                            perturbed_cf_instances[:, i] += torch.randn_like(cf_instances[:, i]) * self.perturb_scale

                    cf_instances.data = perturbed_cf_instances

                    self.optimizer = torch.optim.Adam([cf_instances], lr=learning_rate)

                    iterations = 0
                    prev_loss = float('inf')
                    perturbation_points.append(global_iterations)
                else:
                    break
        
        cf_instances = best_cf_instances
        
        query_out = self.model_interface.predict(query_instance)
        query_out = torch.round(query_out).detach().cpu().numpy()
        cf_out = self.model_interface.predict(cf_instances)
        cf_out = torch.round(cf_out).detach().cpu().numpy()
        query_instance = query_instance.detach().cpu().numpy()
        cf_instances = cf_instances.detach().cpu().numpy()        

        encoded_indices = self.data_interface.encoded_cat_feature_indices

        for feature_indices in encoded_indices:
            cf_instances[:, feature_indices] = np.round(cf_instances[:, feature_indices])
            
            for instance in cf_instances:
                max_idx = feature_indices[np.argmax(instance[feature_indices])]
                for idx in feature_indices:
                    instance[idx] = 1 if idx == max_idx else 0

        query_instance_denorm = self.data_interface.denorm_data(query_instance.copy())
        query_instance_df = pd.DataFrame(query_instance_denorm, columns = self.data_interface.encoded_feature_names)
        cf_instances_denorm = self.data_interface.denorm_data(cf_instances.copy())
        cf_instances_df = pd.DataFrame(cf_instances_denorm, columns = self.data_interface.encoded_feature_names)

        cat_mapping = self.data_interface.get_cat_mapping(cf_instances_df)
        query_instance_df = self.data_interface.onehot_to_cat(query_instance_df, cat_mapping)
        cf_instances_df = self.data_interface.onehot_to_cat(cf_instances_df, cat_mapping)

        cf_instances_df = self.data_interface.apply_feature_precisions(cf_instances_df, self.data_interface.cont_feature_precisions)

        query_instance_df['outcome'] = query_out
        cf_instances_df['outcome'] = cf_out
        
        print('Query instance:', query_instance_df)
        print('CF instances:', cf_instances_df)
        self.plot_loss_curve(loss_values, 'Loss', perturbation_points, iterations_list)
        self.plot_loss_curve(pred_losses, 'Pred Loss', perturbation_points, iterations_list)
        self.plot_loss_curve(proximity_losses, 'Proximity Loss', perturbation_points, iterations_list)
        self.plot_loss_curve(diversity_losses, 'Diversity Loss', perturbation_points, iterations_list)
        self.plot_loss_curve(sparsity_losses, 'Sparsity Loss', perturbation_points, iterations_list)
        self.plot_loss_curve(plausibility_losses, 'Plausibility Loss', perturbation_points, iterations_list)

        return query_instance, cf_instances
    
    def handle_acceptable_ranges(self, acceptable_ranges):
        names, indices, min_vals, max_vals = [], [], [], []
        
        return names, indices, min_vals, max_vals
    
    def get_feature_weights_list(self, query_instance):
        continuous_feature_indices = [self.data_interface.encoded_feature_names.index(name) for name in self.data_interface.cont_feature_names if name in self.data_interface.encoded_feature_names]
        mads_continuous = self.data_interface.get_mads(normalized=True, feature_indices=continuous_feature_indices)
        inverse_mads_continuous = 1 / np.maximum(mads_continuous, 1e-6)
        inverse_mads_continuous = np.round(inverse_mads_continuous, 2)

        self.feature_weights_list = np.ones(query_instance.shape[1])

        for index, weight in zip(continuous_feature_indices, inverse_mads_continuous):
            self.feature_weights_list[index] = weight

        self.feature_weights_list = torch.from_numpy(self.feature_weights_list).float()
        return self.feature_weights_list

    def plot_loss_curve(self, y_values, ylabel, perturbation_points, iterations_list):
        """
        Plots the loss curves.
        """
        plt.figure()
        plt.plot(iterations_list, y_values, label=ylabel)
        for pt in perturbation_points:
            plt.axvline(x=pt, color='gray', linestyle='--', label='Perturbation')
        plt.xlabel('Iterations')
        plt.ylabel(ylabel)
        plt.show()

    def compute_pred_loss(self, cf_instances):
        """
        Computes the first part (prediction) of the loss function.
        """
        pred_loss = 0
        predictions = self.model_interface.predict(cf_instances)
        if self.pred_loss_type == "cross_entropy":
            target = [self.target_cf_class] * len(cf_instances)
            target = torch.tensor(target, dtype=torch.float32)
            target = target.reshape(-1, 1)
            pred_loss = torch.nn.CrossEntropyLoss()(predictions, target)
        elif self.pred_loss_type == "hinge":
            logits = torch.log(torch.abs(predictions - 1e-6) / torch.abs(1 - predictions - 1e-6))
            criterion = torch.nn.ReLU()
            all_ones = torch.ones_like(self.target_cf_class)
            labels = 2 * self.target_cf_class - all_ones
            temp_loss = all_ones - torch.mul(labels, logits)
            pred_loss = criterion(temp_loss)
        return pred_loss.mean()

    def compute_dpp(self, cf_instances):
        """
        Computes the DPP of a matrix of CF instances.
        """
        det_entries = torch.ones(self.total_CFs, self.total_CFs)
        for i in range(self.total_CFs):
            for j in range(self.total_CFs):
                det_entries[i, j] = torch.sum(torch.mul(torch.abs(cf_instances[i] - cf_instances[j]), self.feature_weights_list), dim = 0)
        det_entries = 1.0 / (1.0 + det_entries)
        det_entries += torch.eye(self.total_CFs) * 0.0001
        return torch.det(det_entries)

    def compute_diversity_loss(self, cf_instances):
        """
        Computes the second part (diversity) of the loss function.
        """
        if self.total_CFs == 1:
            return torch.tensor(0.0)
        diversity_loss = self.compute_dpp(cf_instances)
        if diversity_loss < self.diversity_threshold:
            diversity_loss = diversity_loss - (diversity_loss * self.penalty_scale)
        
        return diversity_loss
    
    def compute_distances(self, cf_instances, observed_instances):
        """
        Computes the distances between each counterfactual instance and each observed instance.
        """
        n_cf = cf_instances.size(0)
        observed_instances = torch.tensor(observed_instances.to_numpy(), dtype=torch.float)
        n_observed = observed_instances.size(0)

        cf_expanded = cf_instances.unsqueeze(1).expand(n_cf, n_observed, -1)
        observed_expanded = observed_instances.unsqueeze(0).expand(n_cf, n_observed, -1)

        distances = torch.norm(cf_expanded - observed_expanded, p=1, dim=2)
        return distances

    def compute_plausibility_loss(self, cf_instances, observed_instances, k):
        """
        Computes the third part (plausibility) of the loss function.
        """
        distances = self.compute_distances(cf_instances, observed_instances)
        _, indices = torch.topk(distances, k, largest=False)
        
        k_nearest_distances = torch.gather(distances, 1, indices)
        plausibility_loss = k_nearest_distances.mean()
        if plausibility_loss > self.plausibility_threshold:
           plausibility_loss += plausibility_loss * self.penalty_scale
        return plausibility_loss
        
    def compute_proximity_loss(self, query_instance, cf_instances):
        """
        Computes the proximity part of the loss function.
        """
        proximity_loss = torch.mean(torch.abs(cf_instances - query_instance[0]) * self.feature_weights_list) 

        # Apply threshold and penalty scale
        if proximity_loss > self.proximity_threshold:
            proximity_loss += proximity_loss * self.penalty_scale

        return proximity_loss
    
    def compute_sparsity_loss(self, query_instance, cf_instances, epsilon=torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps))):
        # Initialize a boolean tensor for differences
        differences = torch.zeros_like(cf_instances, dtype=torch.bool)
        
        # For numerical features, check if differences are greater than epsilon
        differences = torch.abs(cf_instances - query_instance[0]) > epsilon
        
        # Count the number of true differences for each counterfactual
        differences = torch.sum(differences, axis=1)
        sparsity = torch.mean(differences.float())
        sparsity_loss = sparsity / len(query_instance[0])
        if sparsity_loss > self.sparsity_threshold:
            sparsity_loss += sparsity_loss * self.penalty_scale
        return sparsity_loss

    def compute_regularization_loss(self, cf_instances):
        """
        Adds a linear equality constraints to the loss functions - to ensure all levels of a categorical variable sums to one
        """
        regularization_loss = 0
        for v in self.data_interface.encoded_cat_feature_indices:
            regularization_loss += torch.sum(torch.pow((torch.sum(cf_instances[:, v[0]:v[-1]+1], axis = 1) - 1.0), 2))

        return regularization_loss

    def compute_loss(self, query_instance, cf_instances, k):
        """
        Computes the overall loss
        """
        pred_loss = self.compute_pred_loss(cf_instances)
        proximity_loss = self.compute_proximity_loss(query_instance, cf_instances,) if self.proximity_weight > 0 else 0.0
        diversity_loss = self.compute_diversity_loss(cf_instances) if self.diversity_weight > 0 else 0.0
        sparsity_loss = self.compute_sparsity_loss(query_instance, cf_instances) if self.sparsity_weight > 0 else 0.0
        plausibility_loss = self.compute_plausibility_loss(cf_instances, self.data_interface.norm_encoded_features, k) if self.plausibility_weight > 0 else 0.0
        regularization_loss = self.compute_regularization_loss(cf_instances)

        loss = pred_loss + (self.proximity_weight * proximity_loss) + (self.sparsity_weight * sparsity_loss) + (self.plausibility_weight * plausibility_loss) + (self.categorical_penalty * regularization_loss) - (self.diversity_weight * diversity_loss) 

        return loss

    def stop_loop(self, iter, loss_diff, cf_instances):
        """
        Determines the stopping condition for optimization.
        """
        if iter < self.min_iter:
            return False

        if iter >= self.max_iter:
            print("Max iterations reached.")
            return True

        if loss_diff <= self.loss_diff_thres:
            self.loss_converge_iter += 1
            if self.loss_converge_iter < self.loss_converge_maxiter:
                return False
            else:
                test_preds = self.model_interface.predict(cf_instances)
                predicted_classes = torch.round(test_preds)

                if (predicted_classes == self.target_cf_class).all():
                    print("All CFs are classified as the desired class.")
                    return True
                else:
                    return False

        return False
