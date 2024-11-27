from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import cohere  # Import the Cohere API

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and dataset scaler
model = tf.keras.models.load_model(r'model.h5')
data_path = r'dyslexia_dataset_with_real_names (1).csv'

# Fit the scaler with the dataset
df = pd.read_csv(data_path)
scaler = StandardScaler()
scaler.fit(df.drop(columns=["Target"]))  # Assuming "Target" is the column with the label

# Initialize Cohere API
cohere_api_key = 'XYZ'  # Replace with your actual API key
co = cohere.Client(cohere_api_key)

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

        # Use Cohere to generate precautions based on the dyslexia probability
        prompt = f"The model has predicted a dyslexia probability of {prob_dyslexia:.2f}. Based on this level of dyslexia, suggest appropriate precautions and next steps."
        cohere_response = co.generate(
            model='command-xlarge-2021-11-22',  # Use an appropriate Cohere model
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7
        )
        precautions = cohere_response.generations[0].text.strip()

        # Return prediction and Cohere response
        result = {
            "probability": f"{prob_dyslexia:.2f}",
            "classification": is_dyslexic,
            "precautions": precautions
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

