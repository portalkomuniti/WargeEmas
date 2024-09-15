import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the data
file_path = 'ADLprediction.csv'  # Update the path if necessary
data = pd.read_csv(file_path)

# Drop missing or invalid data
data = data.dropna()

# Convert categorical columns ('negeri', 'daerah', 'gender') to numerical using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['negeri', 'daerah', 'gender'])

# Drop columns that are not needed for training
columns_to_drop = ['bath', 'dress', 'eating', 'mobility', 'toileting', 'output_one', 'namaklien', 'kp', 'age', 'tarikh_terima', 'tarikh_hantar', 'date', 'dun', 'parlimen']
X = data_encoded.drop(columns=columns_to_drop, errors='ignore')

# Define target variables
y = data_encoded[['bath', 'dress', 'eating', 'mobility', 'toileting']]

# Convert all columns in X to numeric, ignoring any non-numeric errors
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a MultiOutput Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
multi_output_model = MultiOutputClassifier(rf_model)
multi_output_model.fit(X_train, y_train)

# Save the model and the feature names
joblib.dump(multi_output_model, 'multi_output_rf_model_negeri_daerah_gender.pkl')
joblib.dump(X.columns.tolist(), 'feature_names_negeri_daerah_gender.pkl')

# Model evaluation
y_pred = multi_output_model.predict(X_test)

# Convert predictions to a DataFrame for easier handling
y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)

# Calculate evaluation metrics for each output separately
print("Model Performance Metrics for Each Target Variable:")
for column in y_test.columns:
    print(f"\nTarget Variable: {column}")
    print(f"Accuracy: {accuracy_score(y_test[column], y_pred_df[column]):.2f}")
    print(classification_report(y_test[column], y_pred_df[column], zero_division=0))
