from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained linear regression model from the pickle file
model = pickle.load(open('sal_model.sav', 'rb'))

# Load the dataset from the CSV file
data = pd.read_csv('Salary Data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        age = float(request.form['Age'])
        experience = float(request.form['Years of Experience'])
        education = float(request.form['Edu'])
        gender = float(request.form['gen'])

        # Create a feature vector and make a prediction
        features = [age, experience, education, gender]
        prediction = model.predict([features])[0]

        return render_template('result.html', prediction=prediction)
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
