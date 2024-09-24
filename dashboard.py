import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import streamlit as st

# Load Data
@st.cache_data
def load_data():
    sheet_id = '1iCrrT90X3AiRNu5wLkLtlmdsGcoXY8VfDwKWU9hNVYs'
    df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
    return df

df = load_data()

# Title of the Dashboard
st.title('Real-Time Prediction Dashboard: Activity of Daily Live (ADL) for Elderly Individuals')

# Sidebar for parameters
st.sidebar.title("Model Parameters")
max_depth = st.sidebar.slider("Max Depth", 5, 15, 10)
min_samples_split = st.sidebar.slider("Min Samples Split", 0.01, 0.1, 0.01, step=0.01)
max_features = st.sidebar.slider("Max Features", 0.5, 1.0, 0.8, step=0.1)
max_samples = st.sidebar.slider("Max Samples", 0.5, 1.0, 1.0, step=0.1)

# Train-Test Split Function
def train_test_split_and_features(df):
    y = df["target"]
    x = df.drop(['target', 'state', 'district', 'age', 'gender', 'name'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    features = list(x.columns)
    return x_train, x_test, y_train, y_test, features

# Train-Test Split
x_train, x_test, y_train, y_test, features = train_test_split_and_features(df)

# Model Training Function
@st.cache_resource
def fit_and_evaluate_model(x_train, y_train, max_depth, min_samples_split, max_features, max_samples):
    random_forest = RandomForestClassifier(
        random_state=0,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        max_samples=max_samples
    )
    model = random_forest.fit(x_train, y_train)
    return model

model = fit_and_evaluate_model(x_train, y_train, max_depth, min_samples_split, max_features, max_samples)
random_forest_predict = model.predict(x_test)

# Feature Importance Descriptions
feature_importance_descriptions = {
    5: "None",
    4: "None",
    3: "Supervision",
    2: "Assistance with assertive devices",
    1: "Family Assistance",
    0: "Total Assistance"
}

# Tabs in Streamlit
tab1, tab2, tab3 = st.tabs(["Feature Importance", "Model Performance", "Data Overview"])

# Tab 1: Feature Importance
with tab1:
    st.header("Feature Importance")

    # Feature Importance Calculation
    importances = pd.DataFrame(model.feature_importances_)
    importances['features'] = features
    importances.columns = ['importance', 'feature']
    importances.sort_values(by='importance', ascending=True, inplace=True)

    # Plotting Feature Importance
    plt.figure(figsize=(8, 5))
    sns.barplot(x='importance', y='feature', data=importances, color='skyblue')
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    st.pyplot(plt)

    # Display Feature Descriptions
    st.subheader("Feature Importance Levels and Descriptions:")
    for level, description in feature_importance_descriptions.items():
        st.write(f"**{level}**: {description}")

# Tab 2: Model Performance
with tab2:
    st.header("Model Performance")
    st.subheader("Confusion Matrix")
    random_forest_conf_matrix = confusion_matrix(y_test, random_forest_predict)
    st.write(random_forest_conf_matrix)
    
    st.subheader("Accuracy Score")
    random_forest_acc_score = accuracy_score(y_test, random_forest_predict)
    st.write(f"Accuracy: {random_forest_acc_score * 100:.2f}%")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, random_forest_predict))

# Tab 3: Data Overview
with tab3:
    st.header("Data Overview")
    st.write(df.head(10))
    st.write("Data Information:")
    st.write(df.describe())
    st.write("Missing Values:")
    st.write(df.isnull().sum())
    st.write("Target Variable Distribution:")
    st.write(df['target'].value_counts())
