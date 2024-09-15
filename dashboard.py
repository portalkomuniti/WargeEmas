import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set the page configuration for a wide layout
st.set_page_config(layout="wide")

# Load your dataset to extract all unique 'negeri', 'daerah', and 'gender'
file_path = 'ADLprediction.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Calculate the total number of records
total_records = len(data)

# Create a mapping between 'negeri' and 'daerah'
negeri_daerah_mapping = data.groupby('negeri')['daerah'].unique().apply(list).to_dict()

# Extract unique 'negeri' and 'gender' options
negeri_options = sorted(negeri_daerah_mapping.keys())
gender_options = sorted(data['gender'].unique())

# Load the trained model and feature names
model = joblib.load('multi_output_rf_model_negeri_daerah_gender.pkl')
feature_names = joblib.load('feature_names_negeri_daerah_gender.pkl')

# Title of the Dashboard
st.title('Real-Time ADL Prediction Dashboard')

# Sidebar for User Input
st.sidebar.header('Input Features')

# Function to accept user input
def user_input_features():
    # Select negeri
    negeri = st.sidebar.selectbox('Negeri', negeri_options)
    
    # Dynamically update the daerah options based on the selected negeri
    daerah_options = negeri_daerah_mapping[negeri]
    daerah = st.sidebar.selectbox('Daerah', daerah_options)
    
    # Select gender
    gender = st.sidebar.selectbox('Gender', gender_options)

    # Filter data based on the selected daerah
    selected_data = data[(data['negeri'] == negeri) & (data['daerah'] == daerah) & (data['gender'] == gender)]
    record_count = len(selected_data)

    # One-hot encode the input features
    input_data = {'negeri_' + negeri: 1, 'daerah_' + daerah: 1, 'gender_' + gender: 1}

    # Create a DataFrame with all feature names set to 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)

    # Update the DataFrame with the user input
    for key in input_data.keys():
        if key in input_df.columns:
            input_df.at[0, key] = input_data[key]

    return input_df, record_count

# Get user input and record count
input_df, record_count = user_input_features()

# Display the record count for the selected daerah
st.metric(label="Total Records for Selected Daerah and Gender", value=f"{record_count} records")

# Predict with the model whenever user input changes
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

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
