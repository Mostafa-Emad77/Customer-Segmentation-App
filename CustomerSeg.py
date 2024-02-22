import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")  
st.title('Customer Segmentation App')
st.write('This application takes customer data and segments customers based on their Annual Income.')
st.write('You can either upload a CSV file or input the data manually.')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write('Please ensure that the uploaded CSV file contains the following headers: Gender, Age, Annual Income (k$), Spending Score (1-100)')
customer_age = st.number_input('Age', min_value=0)
customer_gender = st.selectbox('Gender', options=['Male', 'Female'])
Annual_Income = st.number_input('Annual Income (k$)', min_value=0)
Spending_Score = st.number_input('Spending Score (1-100)', min_value=1,max_value=100)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
data = None

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    new_data = pd.DataFrame(columns=['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)'])

columns = ['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']

if 'csv_created' not in st.session_state:
    st.session_state.csv_created = False

st.write("If you want to manually input your data, please click the 'Create new csv' button.")
# Add a button to create new_data.csv
if st.button("Create new CSV"):
    # Create an empty DataFrame with your columns
    new_data = pd.DataFrame(columns=columns)
    # Save the empty DataFrame to a CSV file
    new_data.to_csv('new_data.csv', index=False)
    st.success("New CSV is created!")
    # Enable the radio buttons
    st.session_state.csv_created = True


# Add to CSV
if st.button("Add to CSV", disabled=not st.session_state.csv_created):
    # Get the user input
    new_row = {"Age": customer_age, "Gender": customer_gender,
               "Annual Income (k$)": Annual_Income, "Spending Score (1-100)": Spending_Score}
    # Create a DataFrame from the new data
    new_data = pd.DataFrame([new_row])
    # Append the new data to the CSV file
    new_data.to_csv('new_data.csv', mode='a', header=False, index=False)
    st.write("New data added!")

# Clear new_data.csv
if st.button("Clear CSV", disabled=not st.session_state.csv_created):
    # Check if the file exists
    if os.path.isfile('new_data.csv'):
        # Create an empty DataFrame with your columns
        new_data = pd.DataFrame(columns=columns)
        # Overwrite the existing file with the empty DataFrame
        new_data.to_csv('new_data.csv', index=False)
        st.success("Data cleared!")
    else:
        st.warning("File does not exist.")

# Let the user choose which data to use
data_choice = st.radio("Choose which data to use for clustering", ('Uploaded CSV', 'Manually Input Data'), index=0, disabled=not st.session_state.csv_created)

if data_choice == 'Uploaded CSV':
    chosen_data = data
else:
    chosen_data =  pd.read_csv('new_data.csv')

# Preview the chosen data
st.write('Data Preview:')
st.dataframe(chosen_data)

# Create a number input field for the number of clusters
n_clusters = st.number_input('Number of Clusters', min_value=2, value=3)

def preprocess_data(data):
    # Convert 'Gender' column to numerical values
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    return data

def run_kmeans_clustering(data, n_clusters):
    # Selecting features for clustering
    features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Create a KMeans object with the specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    # Fit the KMeans object to the data
    kmeans.fit(features)

    # Get the cluster labels for each data point
    cluster_labels = kmeans.labels_

    return cluster_labels

def plot_clusters(data, labels):
    # Create a scatter plot of 'Annual Income (k$)' vs 'Spending Score (1-100)', colored by cluster labels
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=labels, cmap='viridis')

    # Set the title and labels of the plot
    plt.title('Customer Segmentation')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')

    # Show the plot
    st.pyplot()

def plot_histogram(data, column):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], bins=20, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    st.pyplot()

def plot_elbow_method(data, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.xticks(range(1, max_clusters + 1))
    st.pyplot()

if st.button('Run Clustering'):
    try:
        if chosen_data is None:
            raise ValueError("No data available for clustering")

        # Preprocess the data
        preprocessed_data = preprocess_data(chosen_data)

        # Check if there are any samples available for clustering
        if len(preprocessed_data) == 0:
            raise ValueError("No samples available for clustering")
        if len(preprocessed_data) < 10:
            raise ValueError("Insufficient data points for clustering. Minimum 10 data points required.")
            
        plot_elbow_method(preprocessed_data[['Annual Income (k$)', 'Spending Score (1-100)']])

        # Run the clustering algorithm on the preprocessed data
        cluster_labels = run_kmeans_clustering(preprocessed_data, n_clusters)


        # Plot the clusters
        plot_clusters(preprocessed_data, cluster_labels)

        # Additional plots
        plot_histogram(preprocessed_data, 'Annual Income (k$)')
        plot_histogram(preprocessed_data, 'Spending Score (1-100)')

    except ValueError as e:
        st.warning("Error: " + str(e))
