# lecture_da
Lecture: Introduction to Domain Adaptation

This repository contains two scripts for educational purposes.
1) run_pytorch_basic.py: Standard deep-learning pipeline in Pytorch (used as an example for Empirical Risk Minimization)
2) run_shift_classification.py: Domain adaptation example using DANN algorithm and Importance Weighting (IW). The target dataset is the MNIST-M dataset first proposed in the DANN paper.

Slides are also included in the repo as a pdf file!

# Run code
Make sure to get the necessary libraries like numpy and pytorch:
pip install numpy matplotlib torch torchvision
For importance weighting we use a few extra libraries:
pip install scikit-learn quadprog scipy
