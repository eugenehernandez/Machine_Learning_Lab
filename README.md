# Machine Learning Lab

Statistical machine learning implementations in Python, written during graduate study for the M.S. in Statistical Data Science at Florida State University. Several core algorithms are built from scratch in NumPy (logistic regression, LogitBoost, TISP, FSA, Viterbi, forward-backward and Baum-Welch), and each notebook compares methods on standard high-dimensional benchmarks with train and test error reported side by side.

The emphasis throughout is on understanding how an algorithm behaves, not only whether it scores well. Most notebooks sweep a key setting (tree depth, forest size, boosting rounds, number of selected features, minibatch size) and report where the model starts to overfit.

## Notebooks

| Notebook | Method | Implementation | Data | Headline result |
|---|---|---|---|---|
| [logistic_regression_from_scratch_gradient_ascent](logistic_regression_from_scratch_gradient_ascent.ipynb) | L2-penalized logistic regression trained by gradient ascent on the log likelihood | From scratch (NumPy), benchmarked against scikit-learn | Gisette | 94.1% test accuracy after 300 iterations, versus 96.5% for scikit-learn's solver |
| [logitboost_classifier_gisette_madelon_dexter](logitboost_classifier_gisette_madelon_dexter.ipynb) | LogitBoost with up to 500 boosting rounds | From scratch (NumPy) | Gisette, Dexter, Madelon | Test error falls to 2.5% on Gisette and 8.3% on Dexter, while Madelon shows clear overfitting (training error keeps dropping but test error stays near 42%) |
| [tisp_variable_selection_hard_thresholding](tisp_variable_selection_hard_thresholding.ipynb) | TISP (Thresholding-based Iterative Selection Procedure) with a hard-thresholding penalty | From scratch (NumPy) | Dexter | Tuning the threshold to keep 18 to 1,414 features, test error drops from 25% to 9%, leveling off at about 400 features |
| [fsa_variable_selection_binary_classification](fsa_variable_selection_binary_classification.ipynb) | FSA (Feature Selection with Annealing) for logistic regression | From scratch (NumPy) | Gisette | Test error falls from 17.8% with 10 features to 6.4% with 1,000 features, with train and test error tracking closely |
| [hidden_markov_model_viterbi_baum_welch](hidden_markov_model_viterbi_baum_welch.ipynb) | Hidden Markov model for the "dishonest casino" (fair versus loaded die) | From scratch (Viterbi, forward, backward, Baum-Welch) | Simulated dice rolls | Recovers the most likely hidden state path and re-estimates the transition and emission probabilities from the observed rolls alone |
| [decision_tree_depth_analysis_madelon](decision_tree_depth_analysis_madelon.ipynb) | Decision trees and random forests | scikit-learn | Madelon, Wilt | Best single tree reaches 20.5% test error on Madelon (depth 5). A 100-tree random forest that considers all features at each split reaches 15%, and limiting each split to a random subset of features hurts on this dataset |
| [cnn_regression_image_resolution_prediction](cnn_regression_image_resolution_prediction.ipynb) | Convolutional neural network for regression (predicting an image patch's resolution) | TensorFlow and Keras | About 18,000 training and 2,300 test image patches | Test R² of 0.91 with a minibatch of 32, versus 0.84 with a minibatch of 512, showing the effect of batch size on generalization |
| [gaussian_mixture_model_em_algorithm](gaussian_mixture_model_em_algorithm.ipynb) | Gaussian mixture models fit by EM (initialized with K-means), compared with spectral clustering | scikit-learn | Three two-dimensional clustering datasets | Reports the fitted mixture weights, means, and covariances, and plots the cluster assignments from both methods side by side |
| [label_propagation_usps_handwritten_digits](label_propagation_usps_handwritten_digits.ipynb) | Semi-supervised label propagation compared with a k-nearest-neighbors baseline | scikit-learn | USPS handwritten digits (16 × 16 pixels) | With most labels masked, 1-nearest-neighbor reaches 75.2% test accuracy and label propagation 74.9% |

## Data

The datasets are not included in this repository. Gisette, Dexter, and Madelon come from the NIPS 2003 Feature Selection Challenge and are available from the UCI Machine Learning Repository, as are Wilt and Hill-Valley. USPS is available through `torchvision.datasets`. Each notebook expects the data in a folder next to it, and the file paths near the top of each notebook should be updated to match your machine.

## Running the notebooks

Python 3.9 or later is recommended.

```
pip install -r requirements.txt
jupyter notebook
```

Then open any notebook and run the cells in order.

## Acknowledgments

The TISP implementation adapts ideas from Vasu Gupta's article on gradient ascent for logistic regression and from github.com/arb17b. The LogitBoost plotting style follows a KDnuggets guide to Matplotlib and Seaborn visualization.

## License

Free to copy, modify, or share for academic or research purposes. The code is provided as is, without warranty.
