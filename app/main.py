from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from collections import Counter

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load the trained models
model_ID3 = joblib.load("app/ml/loan_approval_model_ID3.pkl")
model_C45 = joblib.load("app/ml/loan_approval_model_C45.pkl")
model_CART = joblib.load("app/ml/loan_approval_model_CART.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit")
async def submit_loan_application(request: Request,
                                    age: float = Form(...),
                                    income: float = Form(...),
                                    loan_amount: float = Form(...),
                                    credit_score: float = Form(...),
                                    months_employed: float = Form(...),
                                    num_credit_lines: float = Form(...),
                                    interest_rate: float = Form(...),
                                    loan_term: float = Form(...),
                                    dti_ratio: float = Form(...),
                                    education: str = Form(...),
                                    employment_type: str = Form(...),
                                    marital_status: str = Form(...),
                                    has_mortgage: str = Form(...),
                                    has_dependents: str = Form(...),
                                    loan_purpose: str = Form(...),
                                    has_cosigner: str = Form(...)
                                ):
    # Create a DataFrame from the form inputs
    input_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "LoanAmount": [loan_amount],
        "CreditScore": [credit_score],
        "MonthsEmployed": [months_employed],
        "NumCreditLines": [num_credit_lines],
        "InterestRate": [interest_rate],
        "LoanTerm": [loan_term],
        "DTIRatio": [dti_ratio],
        "Education": [education],
        "EmploymentType": [employment_type],
        "MaritalStatus": [marital_status],
        "HasMortgage": [has_mortgage],
        "HasDependents": [has_dependents],
        "LoanPurpose": [loan_purpose],
        "HasCoSigner": [has_cosigner]
    })

    # Make predictions using all three models
    prediction_ID3 = model_ID3.predict(input_data)[0]
    prediction_C45 = model_C45.predict(input_data)[0]
    prediction_CART = model_CART.predict(input_data)[0]

    # Count occurrences of each prediction
    predictions = [prediction_ID3, prediction_C45, prediction_CART]
    prediction_counts = Counter(predictions)

    # Return the prediction that occurred most frequently
    most_common_prediction = max(prediction_counts, key=prediction_counts.get)
    result = "Loan approved" if most_common_prediction == 1 else "Loan denied"
    
    return {"result": result}
