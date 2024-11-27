from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and dataset scaler
model = tf.keras.models.load_model(r'C:\Users\khush\OneDrive\Desktop\ML Hackathon\model.h5')
data_path = r'C:\Users\khush\OneDrive\Desktop\ML Hackathon\dyslexia_dataset_with_real_names (1).csv'

# Fit the scaler with the dataset
df = pd.read_csv(data_path)
scaler = StandardScaler()
scaler.fit(df.drop(columns=["Target"]))  # Assuming "Target" is the column with the label

@app.route('/')
def index():
    return render_template('homepage.html')  # Ensure `homepage.html` contains your HTML structure

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        form_data = request.form.to_dict()
        print("Form Data Received:", form_data)  # Debugging

        # Convert form data to a numeric format
        input_features = {
            "Age_Group": int(form_data.get("Age_Group", 0)),
            "Reading_Trouble": int(form_data.get("Reading_Trouble", 0)),
            "Spelling_Trouble": int(form_data.get("Spelling_Trouble", 0)),
            "Slow_Reading": int(form_data.get("Slow_Reading", 0)),
            "Phonetics_Trouble": int(form_data.get("Phonetics_Trouble", 0)),
            "Memory_Problems": int(form_data.get("Memory_Problems", 0)),
            "Direct_Confusion": int(form_data.get("Direct_Confusion", 0)),
            "Reading_Avoidance": int(form_data.get("Reading_Avoidance", 0)),
            "Speech_Trouble": int(form_data.get("Speech_Trouble", 0)),
            "Family_History": int(form_data.get("Family_History", 0)),
            "Emotional_Impact": int(form_data.get("Emotional_Impact", 0)),
        }

        # Convert input to a DataFrame and scale it
        input_df = pd.DataFrame([input_features])
        scaled_input = scaler.transform(input_df)

        # Model prediction
        probabilities = model.predict(scaled_input)
        prob_dyslexia = probabilities[0][0]  # Assuming the model outputs a single probability
        is_dyslexic = "Dyslexic" if prob_dyslexia > 0.5 else "Not Dyslexic"

        # Return prediction results
        result = {
            "probability": f"{prob_dyslexia:.2f}",
            "classification": is_dyslexic,
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

