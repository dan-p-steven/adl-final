# Humber College Advanced Deep Learning Final Project

## Packages Needed

* optuna
* pandas
* numpy
* scikit-learn
* torch, torchvision, torchaudio with cuda support
* ipykernel

## How to Run
The distilBERT branch was developed on a single ipynb file in notebooks/ directory. The ipynb file can be run under notebooks/.

There are other various ipynb scripts in notebooks/, but they will not run unless you move them to the root directory of the project. These scripts were used to generate visualizations for parts 5 and 6 of the LSTM branch.

The LSTM branch was made using .py files, and it is not recommended to run main.py. This is because main.py was used as the launching script for the various sections of the project as it went on. This means that there is no meaning functionality in the current main.py as it was last used to generate data for the Shapley Values section of the assignment. 

However, it can still be parsed to see functions used during the previous phases of the project.
