# Student Exam Score Prediction Project

This project builds a machine learning model to predict student exam scores based on several performance factors.

## Overview

The project includes:

- **Data**: Student performance data is stored in [data/StudentPerformanceFactors.csv](l:/Student%20Exam%20Score%20Prediction%20project/data/StudentPerformanceFactors.csv).
- **Notebook**: Data exploration, visualization, model training, and evaluation are performed in [natebook.ipynb](l:/Student%20Exam%20Score%20Prediction%20project/natebook.ipynb).
- **Prediction Model**: The model training and prediction scripts are implemented in [predictor.py](l:/Student%20Exam%20Score%20Prediction%20project/predictor.py).
- **Web App**: A Streamlit-based interface is available in [app.py](l:/Student%20Exam%20Score%20Prediction%20project/app.py) for making predictions interactively.

## Requirements

Install the required libraries by running:

```sh
pip install -r requirements.txt
```

## Data

The CSV file, [StudentPerformanceFactors.csv](l:/Student%20Exam%20Score%20Prediction%20project/data/StudentPerformanceFactors.csv), contains various factors such as attendance, hours studied, and scores from previous exams. Data preprocessing (handling missing values and encoding categorical features) is performed in the notebook and predictor script.

## Notebook

The [natebook.ipynb](l:/Student%20Exam%20Score%20Prediction%20project/natebook.ipynb) notebook covers:

- Data loading and cleaning
- Visualization and exploratory data analysis
- Model training using RandomForestRegressor and GradientBoostingRegressor
- Evaluation of model performance (R2 Score and RMSE)
- Feature importance visualization
- Saving the trained model as `student_model.pkl` and `student_model_regression.pkl`

## Prediction Model

The [predictor.py](l:/Student%20Exam%20Score%20Prediction%20project/predictor.py) script:

- Loads the dataset and preprocesses it (dropping missing values and one-hot encoding categorical features)
- Splits the data into training and testing sets
- Trains a RandomForestRegressor with predefined hyperparameters
- Evaluates the model and prints performance metrics
- Saves the trained model for later use

## Running the Project

1. **Train the Model**  
   Run the notebook ([natebook.ipynb](l:/Student%20Exam%20Score%20Prediction%20project/natebook.ipynb)) or the predictor script to train the model and save it:

   ```sh
   python predictor.py
   ```

2. **Make Predictions via Web App (Optional)**  
   Launch the Streamlit app to input new student data and view predictions:
   ```sh
   streamlit run app.py
   ```

## Notes

- Ensure the file paths in the code (e.g., CSV file location) match your local setup.
- Adjust the hyperparameters or preprocessing steps as needed to improve performance.
