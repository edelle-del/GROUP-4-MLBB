import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import altair as alt
from wordcloud import WordCloud
from google.colab import files
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page configuration
st.set_page_config(
    page_title="MLBB Heroes Analysis",
    page_icon="🎮",
    layout="wide"
)

# Title and introduction
st.title("Mobile Legends: Bang Bang E-sports Heroes Stats")
st.markdown("### Group 4 - BM3")
st.markdown("""
- LUMABI, Edelle Gibben
- LUSAYA, John Larence
- PASTIU, Nicholas Rian
- SANTILLAN, Daniel
- VITUG, Sophia
""")


# # Load data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("Mlbb_Heroes.csv")
#     df['Secondary_Role'].fillna('No Secondary Role', inplace=True)
#     return df

# df = load_data()

# # Sidebar
# st.sidebar.header("Navigation")
# page = st.sidebar.selectbox(
#     "Choose a page",
#     ["Dataset Overview", "Role Analysis", "Stats Distribution", "Machine Learning"]
# )

# # Dataset Overview Page
# if page == "Dataset Overview":
#     st.header("Dataset Overview")
    
#     # Display basic dataset information
#     st.subheader("Raw Data")
#     st.dataframe(df)
    
#     st.subheader("Dataset Information")
#     buffer = io.StringIO()
#     df.info(buf=buffer)
#     st.text(buffer.getvalue())
    
#     st.subheader("Statistical Summary")
#     st.write(df.describe())
    
#     st.subheader("Missing Values")
#     st.write(df.isnull().sum())

# # Role Analysis Page
# elif page == "Role Analysis":
#     st.header("Role Analysis")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Primary Role Distribution")
#         fig, ax = plt.subplots(figsize=(8, 6))
#         primary_role_counts = df['Primary_Role'].value_counts()
#         plt.pie(primary_role_counts, labels=primary_role_counts.index, autopct='%1.1f%%', 
#                 colors=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
#         plt.title('Distribution of Primary Roles')
#         st.pyplot(fig)
        
#     with col2:
#         st.subheader("Secondary Role Distribution")
#         fig, ax = plt.subplots(figsize=(8, 6))
#         secondary_role_counts = df['Secondary_Role'].value_counts()
#         plt.pie(secondary_role_counts, labels=secondary_role_counts.index, autopct='%1.1f%%',
#                 colors=['pink', 'purple', 'orange', 'green', 'blue', 'red'])
#         plt.title('Distribution of Secondary Roles')
#         st.pyplot(fig)

# # Stats Distribution Page
# elif page == "Stats Distribution":
#     st.header("Hero Stats Distribution")
    
#     # HP Distribution
#     st.subheader("HP Distribution")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.histplot(df['Hp'], kde=True, bins=10, color='blue')
#     plt.title('Distribution of HP')
#     plt.xlabel('HP')
#     plt.ylabel('Frequency')
#     st.pyplot(fig)
    
#     # Physical Damage Distribution
#     st.subheader("Physical Damage Distribution")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.histplot(df['Phy_Damage'], kde=True, bins=10, color='green')
#     plt.title('Distribution of Physical Damage')
#     plt.xlabel('Physical Damage')
#     plt.ylabel('Frequency')
#     st.pyplot(fig)
    
#     # Correlation Heatmap
#     st.subheader("Correlation Heatmap")
#     fig, ax = plt.subplots(figsize=(12, 8))
#     correlation = df[['Hp', 'Mana', 'Phy_Damage', 'Mag_Damage', 'Phy_Defence', 
#                      'Mag_Defence', 'Mov_Speed']].corr()
#     sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
#     plt.title('Correlation Heatmap for Numerical Variables')
#     st.pyplot(fig)

# # Machine Learning Page
# elif page == "Machine Learning":
#     st.header("Machine Learning Analysis")
    
#     analysis_type = st.selectbox(
#         "Choose Analysis Type",
#         ["Random Forest - Primary Roles", "Random Forest - Secondary Roles", "Supervised Learning"]
#     )
    
#     if analysis_type == "Random Forest - Primary Roles":
#         st.subheader("Random Forest Classification - Primary Roles")
        
#         # Prepare data
#         data = {
#             'Primary_Role': ['Fighter', 'Mage', 'Marksman', 'Tank', 'Assassin', 'Support'],
#             'Count': [33, 25, 18, 16, 13, 9],
#             'Feature 1': [1, 2, 3, 4, 5, 6],
#             'Feature 2': [2, 4, 6, 8, 10, 12],
#             'Feature 3': [3, 6, 9, 12, 15, 18]
#         }
        
#         df_rf = pd.DataFrame(data)
#         feature_colors = {
#             'Feature 1': 'red',
#             'Feature 2': 'green',
#             'Feature 3': 'blue'
#         }
        
#         # Model training and prediction
#         df_rf['Primary_Role_Encoded'], _ = pd.factorize(df_rf['Primary_Role'])
#         X = df_rf[['Feature 1', 'Feature 2', 'Feature 3']]
#         y = df_rf['Primary_Role_Encoded']
        
#         X_resampled, y_resampled = resample(X, y, n_samples=30, random_state=42)
#         X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
#                                                            test_size=0.2, random_state=42)
        
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
        
#         # Display results
#         st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.3f}")
#         st.text("Classification Report:")
#         st.text(classification_report(y_test, y_pred))
        
#         # Feature importance plot
#         fig, ax = plt.subplots(figsize=(12, 6))
#         feature_importances = model.feature_importances_
#         y_pos = range(len(feature_importances))
#         bars = plt.barh(y_pos, feature_importances, align='center')
        
#         for i, bar in enumerate(bars):
#             bar.set_color(feature_colors[X.columns[i]])
#             width = bar.get_width()
#             plt.text(width, bar.get_y() + bar.get_height() / 2,
#                     f'{width*100:.2f}%', ha='left', va='center')
        
#         plt.yticks(range(len(X.columns)), X.columns)
#         plt.xlabel('Feature Importance')
#         plt.title('Random Forest Feature Importance - Primary Roles')
#         st.pyplot(fig)
    
#     elif analysis_type == "Random Forest - Secondary Roles":
#         st.subheader("Random Forest Classification - Secondary Roles")
        
#         # Similar implementation for secondary roles...
#         # [Previous secondary roles analysis code goes here]
        
#     elif analysis_type == "Supervised Learning":
#         st.subheader("Supervised Learning - Secondary Roles")
        
#         # Prepare data
#         data = {
#             'Secondary_Role': ['Support', 'Tank', 'Assassin', 'Mage', 'Fighter', 'Marksman'],
#             'Count': [7, 6, 6, 5, 3, 3]
#         }
        
#         df_sl = pd.DataFrame(data)
#         roles = []
#         for role, count in zip(df_sl['Secondary_Role'], df_sl['Count']):
#             roles.extend([role] * count)
        
#         df_repeated = pd.DataFrame({'Secondary_Role': roles})
        
#         # Model training and prediction
#         vectorizer = TfidfVectorizer()
#         X = vectorizer.fit_transform(df_repeated['Secondary_Role'])
        
#         label_encoder = LabelEncoder()
#         y = label_encoder.fit_transform(df_repeated['Secondary_Role'])
        
#         x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         clf = MultinomialNB()
#         clf.fit(x_train, y_train)
#         y_pred = clf.predict(x_test)
        
#         # Display results
#         st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.3f}")
#         st.text("Classification Report:")
#         st.text(classification_report(y_test, y_pred))

# if __name__ == "__main__":
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("Made with ❤️ by Group 4")
