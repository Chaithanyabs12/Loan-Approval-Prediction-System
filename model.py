import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model():
    df = pd.read_csv("loan_data.csv")

    # Convert categorical values
    df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
    df['Married'] = df['Married'].map({'Yes':1, 'No':0})
    df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

    df = df.fillna(0)

    X = df[['Gender','Married','Education','ApplicantIncome','LoanAmount','Credit_History']]
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model