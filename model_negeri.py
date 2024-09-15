import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
file_path = 'ADLprediction.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Data Preprocessing: Drop missing values
data = data.dropna()

# Feature Engineering: Convert categorical columns to numerical using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['state', 'gender'])

# Select the relevant features for the model (all except the ADL limitations)
columns_to_drop = ['name', 'identification_no', 'district', 'dun', 'parlimen', 'eating', 'bathing', 'dressing', 'toileting', 'mobility']
X = data_encoded.drop(columns=columns_to_drop, errors='ignore')

# Define the target variables (ADL limitations)
y = data_encoded[['eating', 'bathing', 'dressing', 'toileting', 'mobility']]

# Convert all columns in X to numeric and fill NaN values with 0
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
multi_output_model = MultiOutputClassifier(rf_model)

# Train the model on the dataset
multi_output_model.fit(X_train, y_train)

# Save the trained model and the feature names
joblib.dump(multi_output_model, 'multi_output_rf_model_adl.pkl')
joblib.dump(X.columns.tolist(), 'feature_names_adl.pkl')

# Evaluate the trained model
y_pred = multi_output_model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)

# Calculate the performance metrics for the trained model
model_performance = {}
for column in y_test.columns:
    accuracy = accuracy_score(y_test[column], y_pred_df[column])
    report = classification_report(y_test[column], y_pred_df[column], zero_division=0, output_dict=True)
    model_performance[column] = {'accuracy': accuracy, 'classification_report': report}

# Print model performance
print("Model Performance Metrics for Each ADL Category:")
for key, value in model_performance.items():
    print(f"\nADL Category: {key}")
    print(f"Accuracy: {value['accuracy']:.2f}")
    print("Classification Report:", value['classification_report'])
