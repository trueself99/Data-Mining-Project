# Data-Mining-Project
University assigned project where we were given a dataset and asked to apply data mining techniques

## Features
1. **Data Loading and Exploration**:
   - Load data into a pandas DataFrame.
   - Perform initial data exploration by displaying the first few rows of the dataset.

2. **Data Cleaning and Preprocessing**:
   - Handle missing values.
   - Encode categorical variables.
   - Split the data into training and test sets.

3. **Model Building and Training**:
   - Import and use machine learning models from scikit-learn.
   - Train models on the training dataset.
   - Evaluate models using metrics such as accuracy, precision, recall, and F1 score.

4. **Model Evaluation**:
   - Calculate the accuracy of the model on the test set.
   - Plot ROC curves and calculate the Area Under the Curve (AUC) for the model.
   - Create and analyze the confusion matrix.
   - Calculate and display additional metrics like precision, recall, and F1 score.

## Technologies Used
1. **Python**: The programming language used for all the scripting and data analysis.
2. **Pandas**: Used for data manipulation and analysis.
3. **NumPy**: Used for numerical operations on the data.
4. **scikit-learn**: 
   - For model building, training, and evaluation.
   - Includes specific models like Naive Bayes (`GaussianNB`), and functions for metrics like `accuracy_score`, `roc_auc_score`, `roc_curve`, `confusion_matrix`, `precision_score`, `recall_score`, and `f1_score`.
5. **Matplotlib**: For data visualization, particularly for plotting the ROC curve.
