import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("loan_data.csv")

# Convert categorical data
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
df['Married'] = df['Married'].map({'Yes':1, 'No':0})
df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

# Fill missing values
df = df.fillna(0)

# Features & target
X = df[['Gender','Married','Education','ApplicantIncome','LoanAmount','Credit_History']]
y = df['Loan_Status']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("loan_model.pkl", "wb"))

print("✅ Model trained and saved as loan_model.pkl")