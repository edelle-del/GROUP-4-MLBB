# -*- coding: utf-8 -*-
"""BM3 GROUP 4 PROJECT PROPOSAL 2

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17DEuy7XGnMU2hIcqzZBo11P3-W6-fxUE

#MACHINE LEARNING - Mobile Legends: Bang Bang E-sports Heroes Stats
`GROUP 4 - BM3`

1. `LUMABI, Edelle Gibben`
2. `LUSAYA, John Larence`
3. `PASTIU, Nicholas Rian`
4. `SANTILLAN, Daniel`
5. `VITUG, Sophia`
"""

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
from sklearn.preprocessing import LabelEncoder

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

"""# DESCRIBING THE DATASET"""

df = pd.read_csv("Mlbb_Heroes.csv")
df

missing_count = df.isnull().sum()
missing_count

df['Secondary_Role'].fillna('No Secondary Role', inplace=True)

missing_count = df.isnull().sum()
missing_count

print(df.info())

df.describe()

df.shape

df.columns.tolist()

df.nunique()

"""# VALUE COUNTS"""

df['Name'].value_counts()

df['Title'].value_counts()

df['Voice_Line'].value_counts()

df['Release_Date'].value_counts()

df['Primary_Role'].value_counts()

df['Lane'].value_counts()

df['Secondary_Role'].value_counts()

df['Hp'].value_counts()

df['Hp_Regen'].value_counts()

df['Mana'].value_counts()

df['Mana_Regen'].value_counts()

df['Phy_Damage'].value_counts()

df['Mag_Damage'].value_counts()

df['Phy_Defence'].value_counts()

df['Mag_Defence'].value_counts()

df['Mov_Speed'].value_counts()

df['Esport_Wins'].value_counts()

df['Esport_Loss'].value_counts()

"""# EXPLORATORY DATA ANALYSIS (EDA)

## Primary and Secondary Role
"""

#EDA AND PIE CHART FOR PRIMARY ROLE (VITUG, SOPHIA)
data = {
    'Primary_Role': ['Fighter', 'Mage', 'Marksman', 'Tank', 'Assassin', 'Support'],
    'Count': [33, 25, 18, 16, 13, 9]
}

df = pd.DataFrame(data)

#EDA
print("Summary Statistics: ")
print(df.describe())

total_heroes = df['Count'].sum()
print("\nTotal number of heroes:", total_heroes)

print("\nFrequency Distrubution by Role:")
print(df)

#PIE CHART
plt.figure(figsize=(8, 6))
plt.pie(df['Count'], labels=df['Primary_Role'], autopct = '%1.1f%%', colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink'], startangle=90)
plt.title('Distribution of Primary Roles of Heroes in MLBB')
plt.axis('equal')
plt.show()

"""**Primary_Role: Exploratory Data Analysis**

As displayed in this exploratory data analysis, it reveals that the **Fighter** role has the highest count with 33 heroes, while **Support** has the least with 9 heroes, out of the total of 114 heroes. The summary statistics show a mean of **19 heroes per  role**, with **a standard deviation of 8.69**, indicating moderate variability in the distrubution of heroes across each role. In addition, the pie chart provided visualizes the **proportional distrubution of the heroes** of **Mobile Legends: Bang Bang** based on the dataset chosen for this project.
"""

#EDA AND PIE CHART FOR SECONDARY ROLE (VITUG, SOPHIA)
data = {
    'Secondary_Role' : ['No Second Role','Support', 'Tank', 'Assassin', 'Mage', 'Fighter', 'Marksman'],
    'Count': [84, 7, 6, 6, 5, 3, 3]
}

df = pd.DataFrame(data)

#EDA
print("Summary Statistics: ")
print(df.describe())

total_heroes = df['Count'].sum()
print("\nTotal number of heroes with secondary roles:", total_heroes)

print("\nFrequency Distrubution by Secondary Role:")
print(df)

#PIE CHART
plt.figure(figsize=(8, 6))
plt.pie(df['Count'], labels=df['Secondary_Role'], autopct='%1.1f%%',
        colors = ['pink', 'purple', 'orange', 'green', 'blue', 'red'], startangle=90)
plt.title('Distribution of Secondary Roles of Heroes in MLBB')
plt.axis('equal')
plt.show()

"""**Secondary_Role: Findings and Observation**

Based on this exploratory data analysis, it illustrates the distrubution of heroes based on their secondary roles in **Mobile Legends: Bang Bang (MLBB**). It gives vital statistical results such as the mean and standard deviation of the number of heroes per role, revealing that **"Support"** has the most (7), while **"Fighter"** and **"Marksmen"** have the fewest (3 each). The overall number of heroes throughout all secondary roles is 30, and the data is represented by a pie chart, which helps in illustrating the proportionate distribution of each role.

## Distribution of Hp
"""

df = pd.read_csv("Mlbb_Heroes.csv")

import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df['Hp'], kde=True, bins=10, color='blue')
plt.title('Distribution of Hp')
plt.xlabel('Hp')
plt.ylabel('Frequency')
plt.show()

"""This histogram plots the distribution of HP values with a superimposed kernel density estimate (the blue line) to give a representation of the data's frequency. The distribution is right-skewed, meaning it has a longer tail on the higher end of the HP scale. Note most values cluster between 2,250 and 3,000. It shows an apparent peak around 2,500–2,750. Characters that have very low HP, about 1,000–1,250 are relatively few. Thus, there seems to be a pattern where the game or system design does keep most characters' HP in some kind of "sweet spot" near 2,500, while higher or lower values seem less common.

## Distribution of Phy_Damage
"""

plt.figure(figsize=(10, 5))
sns.histplot(df['Phy_Damage'], kde=True, bins=10, color='green')
plt.title('Distribution of Physical Damage')
plt.xlabel('Physical Damage')
plt.ylabel('Frequency')
plt.show()

"""The graph shows physical damage distribution, with a histogram and a superimposed kernel density estimate (green line). The data is roughly normal, peaked around 120, and tapers off to both smaller and higher damage values. Most of the observations fall in the range of 110-130, with fewer cases below 100 and even fewer above 130.

## Frequency of Primary Roles
"""

plt.figure(figsize=(8, 5))
sns.countplot(x='Primary_Role', data=df, palette='Set2')
plt.title('Distribution of Primary Roles')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

"""The graphical representation shows that in this set of characters, there is a leading role which is "Fighter," followed by "Mage," then "Marksman." "Tank" and "Assassin" fall within the middle range, while "Support" is found to be the least. This is a clear leaning towards designs of characters being mostly within the Fighter class.

## Correlation heatmap for numerical variables
"""

plt.figure(figsize=(12, 8))
correlation = df[['Hp', 'Mana', 'Phy_Damage', 'Mag_Damage', 'Phy_Defence', 'Mag_Defence', 'Mov_Speed']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap for Numerical Variables')
plt.show()

"""The heatmap of correlations between different numerical variables features positive correlations as ranges of red and negative correlations as shades of blue. The most important observations include "Physical Defence" and "Movement Speed" with a correlation of 0.42, "Mana" and "Movement Speed" with -0.41, and "HP" with "Physical Defence" correlated with 0.31. Almost all the remaining connections are weak. Thus, a minimal number of linear dependencies between those variable pairs appear to exist.

## Boxplot to analyze Hp by Win/Loss outcome
"""

df['Win'] = df['Esport_Wins'] > df['Esport_Loss']

plt.figure(figsize=(10, 5))
sns.boxplot(x='Win', y='Hp', data=df, palette='coolwarm')
plt.title('Hp vs Win/Loss')
plt.xlabel('Win (1) / Loss (0)')
plt.ylabel('Hp')
plt.show()

"""This boxplot suggests that for those in the "True" category, high values of HP are relatively more common than in the "False" category, and yet there is significant overlap and significant outliers in both groups. This suggests that HP may be only one contributing factor but does not explain the outcome.

## Boxplot for Physical Damage by Win/Loss
"""

plt.figure(figsize=(10, 5))
sns.boxplot(x='Win', y='Phy_Damage', data=df, palette='viridis')
plt.title('Physical Damage vs Win/Loss')
plt.xlabel('Win (True) / Loss (False)')
plt.ylabel('Physical Damage')
plt.show()

"""The boxplot shown suggests some kind of general trend: People under the "False" class receive more Physical Damage than those from the "True" class. Still, there is some overlapping between the two classes, and an outlier for the "True" class illustrates individual variability. So while Physical Damage might play a role, the outcome is certainly not determined by this alone."""

summary_stats = df.describe()
missing_values = df.isnull().sum()

summary_stats, missing_values

"""# MACHINE LEARNING

## Random Forest
In this Python code that uses Random Forest model, it aims to classify the primary roles of heroes based on some hypothetical features and predict the primary role of the hero based on the input features.
"""

#PRIMARY ROLES OF MLBB HEROES CLASSIFICATION USING RANDOM FOREST MODEL

data = {
    'Primary_Role': ['Fighter', 'Mage', 'Marksman', 'Tank', 'Assassin', 'Support'],
    'Count': [33, 25, 18, 16, 13, 9],
    'Feature 1': [1, 2, 3, 4, 5, 6],
    'Feature 2': [2, 4, 6, 8, 10, 12],
    'Feature 3': [3, 6, 9, 12, 15, 18]
}

feature_colors = {
    'Feature 1': 'red',
    'Feature 2': 'green',
    'Feature 3': 'blue'
}

df = pd.DataFrame(data)

#Encode categorical variable
df['Primary_Role_Encoded'],_ = pd.factorize(df['Primary_Role'])

#features and target variable
X = df[['Feature 1', 'Feature 2', 'Feature 3']]
y = df['Primary_Role_Encoded']

#splitting
from sklearn.utils import resample
X_resampled, y_resampled = resample(X,y,
                                    n_samples=30,
                                    random_state=42)

#split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                    test_size=0.2, random_state=42)

#create and train the random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#make predictions
y_pred = model.predict(X_test)

#evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(report)

#visualization of feature importance
plt.figure(figsize=(12, 6))
feature_importances = model.feature_importances_
y_pos = range(len(feature_importances))
bars = plt.barh(y_pos, feature_importances, align='center')

for i, bar in enumerate(bars):
  bar.set_color(feature_colors[X.columns[i]])
  width = bar.get_width()
  plt.text(width, bar.get_y() + bar.get_height() / 2,
           f'{width*100:.2f}',
           ha = 'left',
           va='center',
           fontweight='bold')

plt.yticks(range(len(X.columns)), X.columns, fontsize=12)
plt.xlabel('Feature Importance', fontsize=14, fontweight='bold')
plt.title('Random Forest Feature Importance in Prediction of Features of Primary Roles', fontsize=16, fontweight='bold', pad=20)
plt.grid(axis="x", linestyle="--", alpha=0.7)

legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, label=feature) for feature, color in feature_colors.items()]
plt.legend(handles=legend_elements, loc='lower right', fontsize=12)

plt.tight_layout()
plt.show()

print("\nFeature Importance Percentages:")
for feature, importance in zip(X.columns, feature_importances):
  print(f"{feature}: {importance * 100:.2f}%")

"""On the other hand, this Python code uses Random Forest model to classify the secondary roles of heroes based on some hypothetical features and predict the secondarybrole of the hero based on the input features."""

#SECONDARY ROLES OF MLBB HEROES PREDICTION USING RANDOM FOREST MODEL
data = {
    'Secondary_Role': ['Support', 'Tank', 'Assassin', 'Mage', 'Fighter', 'Marksman'],
    'Count': [7, 6, 6, 5, 3, 3],
    'Feature 1:': [1, 2, 3, 4, 5, 6],
    'Feature 2:': [6, 5, 4, 3, 2, 1],
    'Feature 3': [2, 3, 1, 4, 5, 6]
}

role_colors = {
    'Support': 'blue',
    'Tank': 'green',
    'Assassin': 'red',
    'Mage': 'purple',
    'Fighter': 'orange',
    'Marksman': 'yellow'
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['Secondary_Role_Encoded'] = le.fit_transform(df['Secondary_Role'])

X_columns = ['Feature 1:', 'Feature 2:', 'Feature 3']
X = df[X_columns]
y = df['Secondary_Role_Encoded']

X_resampled, y_resampled = resample(X, y,
                                    n_samples=30,
                                    random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                    test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(report)

plt.figure(figsize=(12, 6))
feature_importances = model.feature_importances_
y_pos = np.arange(len(feature_importances))

bars = plt.barh(y_pos, feature_importances, align='center', color='skyblue')
plt.yticks(y_pos, X_columns, fontsize=12)
plt.xlabel('Feature Importance', fontsize=14, fontweight='bold')
plt.title('Random Forest Feature Importance in Prediction of Features of Secondary Roles', fontsize=16, fontweight='bold', pad=20)
plt.grid(axis="x", linestyle="--", alpha=0.7)

for i, bar in enumerate(bars):
    bar.set_color(list(role_colors.values())[i % len(role_colors)])
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height() / 2,
             f'{width*100:.2f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\nFeature Importance Percentages:")
print("-" * 30)
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance * 100:.2f}%")

"""## Supervised Learning
To be able to input a new champion's stats and have the model predict what secondary role they would fit best in, based on patterns it learned from existing champions.
"""

#USING SUPERVISED LEARNING FOR SECONDARY_ROLE DISTRUBUTION
data = {
    'Secondary_Role': ['Support', 'Tank', 'Assassin', 'Mage', 'Fighter', 'Marksman'],
    'Count': [7, 6, 6, 5, 3, 3]
}

df = pd.DataFrame(data)
roles = []
for role, count in zip(df['Secondary_Role'], df['Count']):
  roles.extend([role] * count)

df_repeated = pd.DataFrame({'Secondary_Role': roles})

print("Repeated Roles:")
print(df_repeated.value_counts())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_repeated['Secondary_Role'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_repeated['Secondary_Role'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(report)