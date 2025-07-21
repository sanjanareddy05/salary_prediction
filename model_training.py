import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('Employee_Salary_Dataset.csv')
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

X = df[['Experience_Years', 'Age', 'Gender']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

with open('salary_model.pkl', 'wb') as f:
    pickle.dump(model, f)
