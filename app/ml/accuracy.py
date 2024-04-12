import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import accuracy_score
import os

# Load data
file_path = "loan_data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found.")

df = pd.read_csv(file_path)

# Preprocessing
X = df.drop(columns=["LoanID", "Default"])  # Features
y = df["Default"]  # Target variable

# Define categorical and numerical columns
categorical_cols = ["Education", "EmploymentType", "MaritalStatus", "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]
numerical_cols = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]

# Define preprocessing steps for categorical and numerical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train Decision Tree classifiers with different algorithms
model_ID3 = Pipeline(steps=[('preprocessor', preprocessor),('classifier', DecisionTreeClassifier(random_state=42, criterion='entropy'))])
model_ID3.fit(X_train, y_train)

model_C45 = Pipeline(steps=[('preprocessor', preprocessor),('classifier', DecisionTreeClassifier(random_state=42, criterion='entropy', splitter='best'))])
model_C45.fit(X_train, y_train)

model_CART = Pipeline(steps=[('preprocessor', preprocessor),('classifier', DecisionTreeClassifier(random_state=42, criterion='gini', splitter='best'))])
model_CART.fit(X_train, y_train)

# Save models
joblib.dump(model_ID3, "loan_approval_model_ID3.pkl")
joblib.dump(model_C45, "loan_approval_model_C45.pkl")
joblib.dump(model_CART, "loan_approval_model_CART.pkl")

print("Models saved successfully.")

# Predictions
y_pred_ID3 = model_ID3.predict(X_test)
y_pred_C45 = model_C45.predict(X_test)
y_pred_CART = model_CART.predict(X_test)

# Calculate accuracy
accuracy_ID3 = accuracy_score(y_test, y_pred_ID3)
accuracy_C45 = accuracy_score(y_test, y_pred_C45)
accuracy_CART = accuracy_score(y_test, y_pred_CART)

print("Accuracy of ID3:", accuracy_ID3)
print("Accuracy of C45:", accuracy_C45)
print("Accuracy of CART:", accuracy_CART)
