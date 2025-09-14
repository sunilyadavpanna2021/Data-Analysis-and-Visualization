
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Load Dataset
data = pd.read_csv('DataAnalyst.csv')

print("Dataset Info:")
print(data.info())
print("\nFirst 5 Rows:")
print(data.head())

# Exploratory Data Analysis (EDA)
print(f"\nDuplicate rows: {data.duplicated().sum()}")
print("\nMissing Values:")
print(data.isnull().sum())
print("\nSummary Statistics:")
print(data.describe(include='all'))

for col in ['Job Title', 'Type of ownership', 'Industry', 'Sector']:
    print(f"\nTop values in {col}:")
    print(data[col].value_counts().head())

# Data Cleaning
data['Rating'].fillna(data['Rating'].median(), inplace=True)

threshold = len(data) * 0.3
data = data.dropna(thresh=threshold, axis=1)

categorical_cols = ['Company Name', 'Industry', 'Sector', 'Type of ownership']
data[categorical_cols] = data[categorical_cols].fillna(method='ffill')

# Extract Salary Information
data['Min Salary'] = data['Salary Estimate'].str.extract(r'\$(\d+)K').astype(float)
data['Max Salary'] = data['Salary Estimate'].str.extract(r'-\s*\$(\d+)K').astype(float)
data['Avg Salary'] = (data['Min Salary'] + data['Max Salary']) / 2
data = data.drop('Salary Estimate', axis=1)

# Feature Engineering
data['Python'] = data['Job Description'].str.contains('Python', case=False, na=False).astype(int)
data['Excel'] = data['Job Description'].str.contains('Excel', case=False, na=False).astype(int)
data['Tech_Skills'] = data['Python'] + data['Excel']

data['City'] = data['Location'].str.split(',', expand=True)[0]
data['State'] = data['Location'].str.split(',', expand=True)[1]

# Statistical Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Model Development
features = ['Rating', 'Tech_Skills', 'Min Salary', 'Max Salary']
X = data[features]
y = data['Avg Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(data['Avg Salary'], bins=20, kde=True)
plt.title("Average Salary Distribution")
plt.xlabel("Average Salary")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Industry', y='Rating', data=data)
plt.xticks(rotation=90)
plt.title("Ratings by Industry")
plt.show()

plt.figure(figsize=(12, 6))
top_jobs = data['Job Title'].value_counts().head(10)
sns.barplot(x=top_jobs.values, y=top_jobs.index)
plt.title("Top 10 Job Titles")
plt.show()

# Streamlit App Code (Optional, run separately using 'streamlit run filename.py')
"""
import streamlit as st

st.title("Data Analyst Job Salary Prediction")

rating = st.slider("Company Rating", 1, 5, 3)
tech_skills = st.slider("Tech Skills Score (Python + Excel)", 0, 2, 1)
min_salary = st.number_input("Minimum Salary (K USD)", min_value=0)
max_salary = st.number_input("Maximum Salary (K USD)", min_value=0)

if st.button("Predict"):
    prediction = model.predict([[rating, tech_skills, min_salary, max_salary]])
    st.write(f"Predicted Average Salary: ${prediction[0]:,.2f}")
"""
    
    
    
    
    


    

        
    

     

 

        
        








    



 

           
           