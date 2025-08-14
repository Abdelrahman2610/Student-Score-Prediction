# Student Score Prediction

This repository contains the implementation of a machine learning project focused on predicting student exam scores using the Student Performance Factors dataset from Kaggle (https://www.kaggle.com/datasets/lainguyn123/student-performance-factors). The project demonstrates advanced data cleaning, visualization, and modeling techniques, including linear regression, polynomial regression, and feature experimentation with categorical encoding.

## Project Overview

**Task**: Build a predictive model for student exam scores based on study hours and other features.  
**Dataset**: StudentPerformanceFactors.csv (sourced from Kaggle, not included in the repository due to size).  
**Objectives**:  
- Clean and preprocess the dataset, including handling missing values with imputation.  
- Visualize relationships between features (e.g., Hours Studied vs. Exam Score) with enhanced plots.  
- Train a linear regression model to predict exam scores.  
- Evaluate model performance using Mean Squared Error (MSE), R-squared (R²), Mean Absolute Error (MAE), and 5-fold cross-validation.  
- Implement polynomial regression (degrees 2 and 3) and compare performance.  
- Experiment with different feature combinations, including encoded categorical variables, to improve predictions.  

**Tools and Libraries**:  
- Python 3.8+  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

## Repository Structure

```
student-score-prediction/
├── notebooks/
│   └── student_score_prediction.ipynb      # Kaggle notebook with code and documentation
├── plots/
│   ├── correlation_heatmap.png             # New: Correlation between numerical features
│   ├── exam_score_histogram.png            # Distribution of exam scores
│   ├── hours_vs_score_scatter.png          # Hours studied vs. exam score with trend line
│   ├── sleep_vs_score_boxplot.png          # Exam score by sleep hours
│   ├── linear_predictions.png              # Linear regression predictions
│   └── polynomial_predictions_degree2.png  # Polynomial regression fit (degree 2)
│   └── polynomial_predictions_degree3.png  # Polynomial regression fit (degree 3)
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation (this file)
```

## Setup and Installation

### Prerequisites
- Git: For cloning the repository (install from https://git-scm.com/downloads).  
- Python 3.8+: For local execution.  
- Jupyter Notebook: For running the notebook locally (pip install jupyter).  
- Kaggle Account: To access the dataset and public notebook.  

### Option 1: Run on Kaggle
1. Visit the public Kaggle notebook: (https://www.kaggle.com/code/abdelrahmansalah2002/student-score-prediction)
2. Ensure the Student Performance Factors dataset is added to Kaggle.
3. Click "Run All" to execute the notebook and view results, including visualizations saved to /kaggle/working/plots/.

### Option 2: Run Locally
1. Clone the repository:  
   ```
   git clone https://github.com/Abdelrahman2610/student-score-prediction.git
   cd student-score-prediction
   ```
2. Install dependencies:  
   ```
   pip install -r requirements.txt
   ```
3. Download the dataset:  
   - Get StudentPerformanceFactors.csv from https://www.kaggle.com/datasets/lainguyn123/student-performance-factors.  
   - Place it in a data/ folder:  
     ```
     mkdir data
     mv ~/Downloads/StudentPerformanceFactors.csv data/
     ```
   - Update the load_data function in notebooks/student_score_prediction.ipynb to use file_path="data/StudentPerformanceFactors.csv".  
4. Run the notebook:  
   ```
   jupyter notebook notebooks/student_score_prediction.ipynb
   ```

## Dependencies
Listed in requirements.txt:  
```
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
```

## Methodology
1. **Data Loading and Cleaning**:  
   - Loaded the dataset using Pandas.  
   - Removed duplicates and imputed missing values (78 in Teacher_Quality, 90 in Parental_Education_Level, 67 in Distance_from_Home) with mode for categoricals and median for numerics.  

2. **Data Visualization**:  
   - Histogram of exam scores to understand the target variable distribution.  
   - Scatter plot of Hours Studied vs. Exam Score with trend lines to explore relationships.  
   - Boxplot of Exam Score by Sleep Hours to investigate additional features.  
   - Correlation heatmap to guide feature selection.  
   - Visualizations are displayed inline and saved to plots/.  

3. **Modeling**:  
   - Linear Regression: Trained on Hours_Studied to predict Exam_Score.  
   - Polynomial Regression: Applied degrees 2 and 3, but no significant improvement over linear.  
   - Feature Experimentation: Tested combinations like Hours_Studied, Sleep_Hours, Attendance, and Tutoring_Sessions, enhanced with one-hot encoding of categorical variables (e.g., Parental_Involvement, Motivation_Level).  

4. **Evaluation**:  
   - Used MSE, R², MAE, and 5-fold cross-validation to assess model performance.  
   - Compared linear, polynomial, and multi-feature models to the baseline.  

5. **Outputs**:  
   - Console: Dataset info, model metrics (MSE, R², MAE, CV R²), and comparisons.  
   - Plots: Saved in plots/ for distribution, relationships, and model predictions.  

## Results
- **Linear Regression**: Baseline model using Hours_Studied (MSE 9.96, R² 0.25).  
- **Polynomial Regression**: Degrees 2 and 3 show no improvement (MSE 9.97, R² 0.25).  
- **Feature Experiments**: Including Attendance and encoded categoricals (e.g., Parental_Involvement) improves performance, with the best model (Hours_Studied, Sleep_Hours, Attendance, Tutoring_Sessions + encoding) achieving MSE 3.68, R² 0.72, MAE 1.06, CV R² 0.63 (±0.16).  
- **Visualizations**: Plots in plots/ illustrate data distributions, feature relationships, and model fits, including the correlation heatmap.  

See notebooks/student_score_prediction.ipynb for detailed outputs and metrics.

## Future Improvements
- Experiment with additional models like Random Forest or XGBoost for potentially higher accuracy.  
- Perform feature selection to identify the most impactful predictors.  
- Explore advanced imputation techniques or feature engineering for missing data.  

## Author
Abdelrahman Mohamed Salah  
GitHub: https://github.com/Abdelrahman2610  
Kaggle: https://www.kaggle.com/abdelrahmansalah2002 

## Acknowledgments
- Dataset provided by Kaggle (https://www.kaggle.com/datasets/lainguyn123/student-performance-factors).  
- Tools: Python, Pandas, Matplotlib, Seaborn, Scikit-learn.  
