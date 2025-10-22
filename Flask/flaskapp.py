from flask import Flask, render_template, request
import pandas as pd
import joblib as jb

app = Flask(__name__)

# Load trained model
model = jb.load('../models/SalesModel.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    tv, radio, newspaper = [float(x) for x in request.form.values()]
    new_data = pd.DataFrame({'TV': [tv], 'radio': [radio], 'newspaper': [newspaper]})
    result = model.predict(new_data)
    return render_template('index.html', prediction_text=f'Predicted Sales: {result[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
