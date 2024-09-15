import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set the page configuration for a wide layout
st.set_page_config(layout="wide")

# Load the trained model and feature names
model = joblib.load('multi_output_rf_model_adl.pkl')
feature_names = joblib.load('feature_names_adl.pkl')

# Load your dataset to extract unique 'state' and 'gender'
file_path = 'ADLprediction.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Extract unique 'state' options
state_options = sorted(data['state'].unique())

# Title of the Dashboard
st.title('Real-Time ADL Prediction Dashboard for Elderly Individuals')

# Sidebar for User Input
st.sidebar.header('Input Features')

# Function to accept user input
def user_input_features():
    # Select state
    state = st.sidebar.selectbox('State', state_options)

    # Filter dataset based on the selected state to get unique gender options
    gender_options = sorted(data[data['state'] == state]['gender'].unique())
    
    # Select gender based on filtered options
    gender = st.sidebar.selectbox('Gender', gender_options)

    # Create an empty DataFrame with all feature names set to 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)

    # One-hot encode the input features
    input_data = {'state_' + state: 1, 'gender_' + gender: 1}
    
    # Update the DataFrame with user input
    for key in input_data.keys():
        if key in input_df.columns:
            input_df.at[0, key] = input_data[key]

    # Ensure all columns expected by the model are present
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Check for missing or unexpected values
    input_df = input_df.fillna(0)

    return input_df, state

# Get user input DataFrame and the selected state
input_df, selected_state = user_input_features()

# Correctly calculate the record count for the selected state
record_count = len(data[data['state'] == selected_state])

# Display the record count for the selected state
st.metric(label="Total Records for Selected State", value=f"{record_count} records")

# Predict with the model whenever user input changes
try:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
    st.stop()

# Prepare data for display
activities = ['Eating', 'Bathing', 'Dressing', 'Toileting', 'Mobility']
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

# Arrange the prediction results horizontally without "Needs Assistance" label
st.subheader('Detailed Prediction Results')
cols = st.columns(len(activities))
for i, activity in enumerate(activities):
    percentage = round(needs_assistance[i], 2)
    with cols[i]:
        st.metric(label=activity, value=f"{percentage} %")
