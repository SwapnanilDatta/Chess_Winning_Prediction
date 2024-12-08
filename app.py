from flask import Flask, request, jsonify,redirect,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


turns_model = pickle.load(open('turns_model.pkl', 'rb'))
winner_model = pickle.load(open('winner_model.pkl', 'rb'))


le = pickle.load(open('label_encoder.pkl', 'rb')) 
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.json
        white_rating = int(data['white_rating'])
        black_rating = int(data['black_rating'])
        opening_name = data['opening_name']
        
        # Encode opening_name
        opening_name_encoded = le.transform([opening_name])[0]
        
        # Create feature array
        features = np.array([[white_rating, black_rating, opening_name_encoded]])
        
        # Make predictions
        turns_prediction = turns_model.predict(features)[0]
        winner_proba = winner_model.predict_proba(features)[0]  # Probabilities for draw, white, black
        
        # Prepare response
        response = {
            'predicted_turns': turns_prediction,
            'winner_probabilities': {
                'draw': winner_proba[2],
                'white': winner_proba[1],
                'black': winner_proba[0]
            }
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
