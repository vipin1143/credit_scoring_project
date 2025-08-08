from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Load All Model Assets ---
try:
    model = joblib.load('lgbm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    training_columns = joblib.load('training_columns.joblib')
    print("✅ Model, scaler, and training columns loaded successfully!")
except FileNotFoundError as e:
    print(f"🔴 CRITICAL ERROR: Could not find a required model file. {e}")
    model, scaler, training_columns = None, None, None
except Exception as e:
    print(f"🔴 An unexpected error occurred while loading model assets: {e}")
    model, scaler, training_columns = None, None, None

# --- Scorecard Function ---
def probability_to_score(prob_of_default):
    """Converts probability to a 300-900 score."""
    score = 300 + (600 * (1 - prob_of_default))
    return int(score)

# --- API Endpoints ---
@app.route('/status', methods=['GET'])
def status():
    """A simple endpoint to check if the API is running."""
    if model and scaler and training_columns:
        return jsonify({'status': 'ok', 'message': 'API is online and all assets are loaded.'})
    else:
        return jsonify({'status': 'error', 'message': 'API is running, but some assets failed to load.'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """The main endpoint to make credit score predictions."""
    if not all([model, scaler, training_columns]):
        return jsonify({'error': 'Model is not loaded properly. Cannot make predictions.'}), 500

    try:
        data = request.get_json()
        input_df = pd.DataFrame(data, index=[0])
        input_df_reordered = input_df[training_columns]
        input_scaled = scaler.transform(input_df_reordered)
        prob_default = model.predict_proba(input_scaled)[:, 1][0]
        credit_score = probability_to_score(prob_default)
        
        return jsonify({
            'probability_of_default': float(prob_default),
            'credit_score': credit_score
        })
    except Exception as e:
        print(f"🔴 Prediction Error: {e}")
        return jsonify({'error': f'Prediction failed: {e}'}), 500

if __name__ == '__main__':
    # --- THE FIX: Disable the reloader to prevent timeouts ---
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
