import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import streamlit as st

sheet_id = '1CcszXiJpeVR7T0cj5ZfFxyy6cnu8MHWnhMe1KMqBfA0'

df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")

print(df.head())

print(f'Data Shape: {df.shape}')

# Example: Convert categorical columns to numeric using LabelEncoder
categorical_columns = ['state', 'district', 'gender', 'name', 'target']
for column in categorical_columns:
    df[column] = LabelEncoder().fit_transform(df[column])

# Assuming your target variable is 'state' and using all other columns as features
X = df.drop(columns=['state'])  # Features
y = df['state']                 # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

import streamlit as st

# Streamlit app
st.title('Random Forest Classifier Prediction')

# Create input fields for user input based on the features
input_data = []
for column in X.columns:
    input_value = st.text_input(f'Enter value for {column}')
    input_data.append(float(input_value))

# If the user clicks the 'Predict' button
if st.button('Predict'):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=X.columns)
    
    # Make predictions
    prediction = rf_classifier.predict(input_df)
    
    # Display the prediction
    st.write(f'Prediction: {prediction[0]}')

# Run the app
if __name__ == '__main__':
    st.run()
