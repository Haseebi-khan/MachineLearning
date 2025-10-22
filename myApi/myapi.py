from flask import Flask, request, jsonify
import pandas as pd
import joblib as jb

app = Flask(__name__)

# Load the trained model
model = jb.load('../models/SalesModel.pkl')

# Create API routing call
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON Request
    new_data = request.json
    
    # Convert JSON request to Pandas DataFrame
    df = pd.DataFrame(new_data)
    
    # Get prediction
    prediction = model.predict(df)
    
    # Return JSON version of Prediction
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=12345)
    
    
    
# download Postman locally and past below running http
# http://127.0.0.1:12345/predict

# use post method and provide below Data_features
# [
# {
#   "TV": 230.1,
#   "radio": 37.8,
#   "newspaper": 69.2
# }
# ]