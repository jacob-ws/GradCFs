# Flexible-CFX: Flexible and Interpretable Counterfactual Explanation Methods

This repository contains the official implementation of DiPACE and GradCFA, two methods developed to generate flexible, interpretable, and plausible counterfactual explanations for machine learning models using gradient-descent.

These methods were developed as part of early PhD research exploring how to improve interpretability and flexibility in counterfactual explanations.

---

## Method Overview:

### DiPACE: Diverse and Plausible Counterfactual Explanations

- Published at ICAART 2025
- Selected and accepted for post-publication in Lecture Notes in Computer Science (accepted: 01/08/2025, publication expected in December 2025).

DiPACE uses a weighted sum loss function including objectives for:
- validity: encouraging correctly classified CFs (hinge loss minimisation)
- diversity: encouraging multiple distinct CFs per instance (DPP distance maximisation)
- plausibility: encouraging CFs which align with the data distribution (kNN distance minimisation)
- proximity: encouraging CFs with minimal distance between the original and CF values (L1 distance minimisation)
- sparsity: encouraging CFs with minimal features changing (hamming distance minimisation)


### GradCFA: Gradient-Based Counterfactual and Feature Attribution Explanations

- Published in IEEE Transactions on Artificial Intelligence (18/03/2025)

GradCFA builds on DiPACE by integrating feature attribution through gradient magnitudes, providing insight into the influence of each feature in counterfactual generation.

---

### Citation:

If you use this code in your research, please consider citing:

@inproceedings{sanderson2025dipace,
    author = {Sanderson, Jacob and Mao, Hua and Woo, Wai Lok},
    title = {DiPACE: Diverse, Plausible and Actionable Counterfactual Explanations},
    booktitle = {Proceedings of the 17th International Conference on Agents and Artificial Intelligence},
    year = {2025}
}

@article{sanderson2025gradcfa,
  title={GradCFA: A Hybrid Gradient-Based Counterfactual and Feature Attribution Explanation Algorithm for Local Interpretation of Neural Networks},
  author={Sanderson, Jacob and Mao, Hua and Woo, Wai Lok},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2025},
  publisher={IEEE},
  volume={Early Access},
  pages={1--13},
  DOI={10.1109/TAI.2025.3552057}
}
