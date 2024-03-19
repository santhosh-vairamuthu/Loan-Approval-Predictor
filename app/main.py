from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load the trained model
model = joblib.load("app/ml/loan_approval_model.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit")

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


    input_data = np.array([[age, income, loan_amount, credit_score, months_employed, num_credit_lines,
                            interest_rate, loan_term, dti_ratio, education, employment_type,
                            marital_status, has_mortgage, has_dependents, loan_purpose, has_cosigner]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Render prediction result
    result = "Loan approved" if prediction == 1 else "Loan denied"
    return {"result": result}