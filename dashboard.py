import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set the page configuration for a wide layout
st.set_page_config(layout="wide")

# Load the trained model and feature names
model = joblib.load('multi_output_rf_model_adl.pkl')
feature_names = joblib.load('feature_names_adl.pkl')

# Load your dataset to extract unique 'negeri'
file_path = 'ADLprediction.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Extract unique 'negeri' options
negeri_options = sorted(data['negeri'].unique())

# Title of the Dashboard
st.title('Real-Time Activity of Daily Living (ADL) Prediction Dashboard for Older Individuals')

# Sidebar for User Input
st.sidebar.header('Input Features')

# Function to accept user input
def user_input_features():
    # Select negeri
    negeri = st.sidebar.selectbox('Negeri', negeri_options)

    # Enter additional user inputs like age
    age = st.sidebar.slider('Age', 60, 100, 70)  # Elderly age range

    # Create an empty DataFrame with all feature names set to 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)

    # One-hot encode the input features
    input_data = {'negeri_' + negeri: 1, 'age': age}
    
    # Update the DataFrame with user input
    for key in input_data.keys():
        if key in input_df.columns:
            input_df.at[0, key] = input_data[key]

    # Ensure all columns expected by the model are present
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Check for missing or unexpected values
    input_df = input_df.fillna(0)

    return input_df, negeri

# Get user input DataFrame and selected negeri
input_df, selected_negeri = user_input_features()

# Correctly calculate the record count for the selected negeri
record_count = len(data[data['negeri'] == selected_negeri])

# Display the record count for the selected negeri
st.metric(label="Total Records for Selected Negeri", value=f"{record_count} records")

# Predict with the model whenever user input changes
try:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
    st.stop()

# Prepare data for display
activities = ['Bath', 'Dress', 'Eating', 'Mobility', 'Toileting']
needs_assistance = [prediction_proba[i][0][1] * 100 for i in range(len(activities))]

# Identify the highest and lowest probability
highest_index = needs_assistance.index(max(needs_assistance))
lowest_index = needs_assistance.index(min(needs_assistance))

# Display scorecards for highest and lowest probabilities
st.subheader('Summary of Assistance Needs')
col_high, col_low = st.columns(2)

with col_high:
    st.metric(label=f"Highest Needs Assistance: {activities[highest_index]}",
              value=f"{round(needs_assistance[highest_index], 2)}%",
              delta="High")

with col_low:
    st.metric(label=f"Lowest Needs Assistance: {activities[lowest_index]}",
              value=f"{round(needs_assistance[lowest_index], 2)}%",
              delta="Low")

# Display the bar chart below the summary
st.subheader('Probability of Needs Assistance for Each Activity')
fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size to fit the screen better
ax.bar(activities, needs_assistance, color='skyblue')
ax.set_xlabel('Activities')
ax.set_ylabel('Probability of Needs Assistance (%)')
ax.set_title('Probability of Needs Assistance')
st.pyplot(fig)

# Arrange the prediction results horizontally
st.subheader('Detailed Prediction Results')
cols = st.columns(len(activities))
for i, activity in enumerate(activities):
    result = "Needs assistance" if prediction[0][i] else "No assistance needed"
    percentage = round(needs_assistance[i], 2)
    with cols[i]:
        st.metric(label=activity, value=result, delta=f"{percentage} %")
