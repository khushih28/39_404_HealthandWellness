from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler
import cohere

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and dataset scaler
model_path = os.path.join('models', 'model.h5')
data_path = os.path.join('data', 'dyslexia_dataset_with_real_names (1).csv')
model = tf.keras.models.load_model(model_path)

# Load the dataset for reference
df = pd.read_csv(data_path)



cohere_api_key = '8L9hBMX89WYoSnigKMAdfcS3eL18e1LUZqtBIPSP'  # Replace with your actual API key
co = cohere.Client(cohere_api_key)

@app.route('/')
def index():
    return render_template('homepage.html')  # Ensure homepage.html contains your HTML structure

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        form_data = request.form.to_dict()
        print("Form Data Received:", form_data)  # Debugging

        # Convert form data to a numeric format (1 for "Yes", 0 for "No")
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

        # Convert input to a DataFrame
        input_df = pd.DataFrame([input_features])

    

        # Model prediction
        probabilities = model.predict(input_df)
        prob_dyslexia = probabilities[0][0]  # Assuming the model outputs a single probability
        is_dyslexic = "Dyslexic" if prob_dyslexia > 0.5 else "Not Dyslexic"

        print("Prediction:", is_dyslexic, "with probability:", prob_dyslexia)

        # Generate text response (optional, based on model prediction)
        prompt = f"The model has predicted a dyslexia probability of {prob_dyslexia:.2f}. Based on this level of dyslexia, suggest appropriate precautions and next steps."
        cohere_response = co.generate(
            model='command-r-08-2024',  # Use an appropriate Cohere model
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7
        )
        precautions = cohere_response.generations[0].text.strip()

        # Return prediction results and precautions
        result = {
            "probability": f"{prob_dyslexia:.2f}",
            "classification": is_dyslexic,
            "precautions": precautions,
        }
        print("Result:", result)  # Debugging
        return jsonify(result)

    except Exception as e:
        # Log the error and return a friendly message
        print(f"Error: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request."}), 400

if __name__ == '__main__':
    app.run(debug=True)