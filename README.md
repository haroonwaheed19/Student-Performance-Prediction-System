🎓 Student Performance Analysis: Regression & Classification

This project analyzes student performance using the Portuguese student dataset and applies machine learning techniques to predict grades (regression) and classify pass/fail outcomes (classification). It also explores hyperparameter tuning using GridSearchCV for performance optimization.

📁 Dataset

    Source: UCI Machine Learning Repository

    File Used: student-mat.csv

    Delimiter: Semicolon (;)

    Target Variables:

        G3 (Final Grade, used in regression)

        pass (Binary label for classification: 1 if G3 ≥ 10, else 0)

📌 Objectives

    Preprocess categorical and numerical features.

    Predict final grades using:

        Linear Regression

        Random Forest Regressor

    Classify students into pass/fail categories.

    Visualize results (scatter plots, confusion matrix, feature importance).

    Optimize models using GridSearchCV.

🛠️ Technologies & Libraries

    Python 3.x

    pandas, numpy

    seaborn, matplotlib

    scikit-learn:

        LinearRegression, RandomForestRegressor, RandomForestClassifier

        LabelEncoder, StandardScaler

        train_test_split, GridSearchCV

        classification_report, confusion_matrix, mean_squared_error, r2_score

📊 Data Preprocessing

    Handled missing values (none in dataset).

    Applied LabelEncoder to convert categorical variables.

    Standardized features using StandardScaler.

    Created binary pass label based on G3.

📈 Regression Models

1. Linear Regression

    Evaluated using:

        MSE, RMSE, MAE

        R² Score

2. Random Forest Regressor

    Parameters:

        n_estimators=100

        max_depth=10

        min_samples_leaf=5

📌 Results Visualized:

    Actual vs Predicted scatter plots

    Top 15 Feature Importances

📉 Classification Model

✅ Random Forest Classifier

    Target: pass (0 = Fail, 1 = Pass)

    Metrics:

        Accuracy

        Classification Report (Precision, Recall, F1-score)

        Confusion Matrix (with heatmap)

⚙️ Hyperparameter Tuning (GridSearchCV)

    Regressor:

        Parameters tuned:

            n_estimators, max_depth, min_samples_leaf

        Scoring: neg_mean_squared_error

    Classifier:

        Parameters tuned:

            n_estimators, max_depth, min_samples_leaf

        Scoring: accuracy

📷 Visual Outputs

    📌 Scatter plots for actual vs predicted grades (Linear & Random Forest)

    📌 Bar plot for feature importance

    📌 Confusion matrix for classification results

🧠 Future Improvements

    Try advanced models (XGBoost, Gradient Boosting)

    Add cross-validation visualization

    Analyze student-por.csv dataset similarly

    Implement deep learning approaches

📝 License

This project is open source and available under the MIT License.

🙌 Acknowledgements

    UCI Machine Learning Repository

    scikit-learn team

    matplotlib & seaborn creators
