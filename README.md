# Loan Approval Predictor

This is a simple loan approval predictor application built using FastAPI and scikit-learn. It predicts whether a loan application will be approved or denied based on input features such as age, income, loan amount, etc.

## Features

- **Frontend**: HTML form with Bootstrap for styling.
- **Backend**: FastAPI for serving predictions.
- **Machine Learning Model**: Decision Tree Classifier trained using scikit-learn.
- **Data**: CSV dataset containing loan application information.

## Installation

1. Clone the repository:

```
git clone https://github.com/santhosh-vairamuthu/Loan-Approval-Predictor

```

2. Install dependencies:

```
pip install -r requirements.txt

```

3. Run the FastAPI application:

```
uvicorn app.main:app --reload
```


## Usage

1. Open your web browser and go to `http://localhost:8000` to access the loan application form.
2. Fill out the form with the required information.
3. Submit the form to get the loan approval prediction result.

## Dataset

The dataset used for training the model is stored in `app/ml/loan_data.csv`. It contains information about loan applications, including age, income, loan amount, credit score, etc.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


