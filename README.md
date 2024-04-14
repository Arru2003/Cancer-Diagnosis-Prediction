# Cancer-Prediction

## Overview
This repository contains code for building and evaluating machine learning models to predict breast cancer diagnosis based on various features extracted from diagnostic images. The models implemented here utilize popular algorithms such as Support Vector Machine (SVM), Random Forest, and XGBoost.

## Requirements
Ensure you have the following dependencies installed:
- Python 3.x
- Jupyter Notebook or any Python IDE
- Required Python packages: numpy, pandas, seaborn, scikit-learn, xgboost, matplotlib, imbalanced-learn (imblearn)

## Usage
1. Clone this repository to your local machine.
2. Ensure all required dependencies are installed (see 'Requirements' section).
3. Download the dataset data.csv and place it in the same directory as the code files.
4. Open the Jupyter Notebook or Python script containing the code.
5. Run the code cells or execute the script to train the models and perform predictions.
6. Adjust parameters or explore different algorithms as needed.

## Code Structure
- data.csv: Dataset containing the features and diagnosis labels.
- Cancer_pred1.ipynb: Jupyter Notebook containing the main code for data preprocessing, model training, evaluation, and visualization.
- README.md: This file, providing an overview of the project and instructions for usage.

## Functionality
- *Data Preprocessing*: The code performs initial data exploration, handling missing values, encoding categorical variables, and scaling numerical features.
- *Exploratory Data Analysis (EDA)*: Visualizations such as heatmaps and box plots are generated to understand the data distribution and identify correlations between features.
- *Model Training*: Support Vector Machine (SVM), Random Forest, and XGBoost classifiers are trained using grid search for hyperparameter tuning.
- *Model Evaluation*: The models are evaluated using metrics such as accuracy, confusion matrix, and classification reports.
- *Feature Importance*: Feature importance is assessed using Random Forest, and a bar plot is generated to visualize the importance of each feature.
- *Handling Imbalanced Data*: Synthetic Minority Over-sampling Technique (SMOTE) is used to address class imbalance before training models.
- *Saving and Loading Models*: Trained models are saved using pickle for future use, and examples of loading and making predictions are provided.

## Author
- Arru2003
  
## Acknowledgments
- Inspiration, code snippets, or resources that you found helpful and utilized in this project.
