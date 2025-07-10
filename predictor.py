# ------------------ Import Libraries ------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ Load Data ------------------
df = pd.read_csv('L:\\Student Exam Score Prediction project\\data\\StudentPerformanceFactors.csv')


# ------------------ Handle Missing Values ------------------
df = df.dropna() 

# ------------------ Encode Categorical Features ------------------
categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                        'Motivation_Level', 'Internet_Access', 'Teacher_Quality',
                        'Peer_Influence', 'Learning_Disabilities']

df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# ------------------ Select Top Important Features ------------------
important_features = ['Attendance', 'Hours_Studied', 'Previous_Scores',
                      'Tutoring_Sessions', 'Sleep_Hours', 'Physical_Activity']


encoded_cols = [col for col in df_encoded.columns if any(keyword in col for keyword in ['Involvement', 'Resources', 'Motivation', 'Peer', 'Teacher', 'Learning'])]
important_features += encoded_cols

# ------------------ Define X & y ------------------
X = df_encoded[important_features]
y = df_encoded['Exam_Score']

# ------------------ Split Data ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Random Forest Model with Hyperparameters ------------------
rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results:")
print("R2 Score:", round(r2_score(y_test, rf_pred), 4))
rmse = mean_squared_error(y_test, rf_pred)
print("RMSE:", round(rmse ** 0.5, 4))

# ------------------ Gradient Boosting Model ------------------
gb_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

print("\n Gradient Boosting Results:")
print("R2 Score:", round(r2_score(y_test, gb_pred), 4))
rmse = mean_squared_error(y_test, rf_pred)
print("RMSE:", round(rmse ** 0.5, 4))

# ------------------ Feature Importance Visualization ------------------
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(12, 5))
plt.title("Feature Importance (Random Forest)")
plt.show()

# ------------------ Predict Example Student ------------------
example_student = pd.DataFrame([{
    'Attendance': 85,
    'Hours_Studied': 40,
    'Previous_Scores': 65,
    'Tutoring_Sessions': 2,
    'Sleep_Hours': 7,
    'Physical_Activity': 4,
    # Example of one-hot columns (set 0 or 1 as per your model):
    'Parental_Involvement_Low': 0,
    'Access_to_Resources_Low': 0,
    'Motivation_Level_Low': 1,
    'Peer_Influence_Positive': 1,
    'Teacher_Quality_Low': 0,
    'Learning_Disabilities_Yes': 0,
    # Add all other one-hot encoded columns if present
}])

# Fill missing columns in the example input (just for demo)
for col in X.columns:
    if col not in example_student.columns:
        example_student[col] = 0

example_student = example_student[X.columns]  # Reorder columns to match training data

predicted_score = rf_model.predict(example_student)

print("\nPredicted Exam Score for Example Student:", round(predicted_score[0], 2))


import pickle

# Save the RandomForest model
with open('student_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

print("\nModel saved as 'student_model.pkl'")
