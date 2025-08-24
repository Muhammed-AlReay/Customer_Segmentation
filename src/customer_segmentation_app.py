import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🔹 Customer Segmentation App (K-Means)")
st.write("Upload your customer CSV file to segment them using K-Means clustering.")

# File uploader
uploaded_file = st.file_uploader("⬆️ Upload your customer CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show raw columns
    st.write("🧾 Raw column names in your file:", df.columns.tolist())

    # Normalize column names
    df.columns = [col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") for col in df.columns]

    # Rename known columns to standard format
    column_mapping = {
        'gender': 'Gender',
        'age': 'Age',
        'annual_income_k$': 'Annual Income (k$)',
        'annual_income': 'Annual Income (k$)',
        'spending_score_1_100': 'Spending Score'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Required columns after renaming
    required_cols = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score']

    if all(col in df.columns for col in required_cols):
        df = df[required_cols]

        # Encode gender
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'male': 0, 'female': 1})

        st.subheader("✅ Cleaned Data Preview")
        st.dataframe(df.head())

        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # Sidebar for cluster count
        k = st.sidebar.slider("Select number of clusters (K):", 2, 10, 5)

        # Apply KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)

        # Scatter plot
        st.subheader("📊 Cluster Visualization")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score', hue='Cluster', palette='tab10', s=100, ax=ax)
        plt.title("Customer Segmentation")
        st.pyplot(fig)

        # Cluster Summary
        st.subheader("📈 Cluster Summary")
        st.dataframe(df.groupby('Cluster').mean())

        # Elbow Method
        st.subheader("🦴 Elbow Method")
        inertia = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=42)
            km.fit(scaled_data)
            inertia.append(km.inertia_)

        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, 11), inertia, marker='o')
        ax2.set_title("Elbow Method")
        ax2.set_xlabel("Number of Clusters")
        ax2.set_ylabel("Inertia")
        st.pyplot(fig2)

        # Download segmented file
        st.subheader("📥 Download Segmented Data")
        st.download_button("📤 Download CSV", df.to_csv(index=False), file_name="segmented_customers.csv", mime="text/csv")

    else:
        st.error(f"⚠️ Your file must contain the following columns: {required_cols}")
else:
    st.info("👆 Please upload a CSV file to begin.")
