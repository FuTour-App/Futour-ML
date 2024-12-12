from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

app = Flask(__name__)


MODEL_PATH = "model/image_genre_classifier_v3.keras"
model = load_model(MODEL_PATH)

recommendation_data = pd.read_json('./json/cleaned_osm_data_described.json')

class_labels = ['adventure',
 'art',
 'beach',
 'historical',
 'monument',
 'museum',
 'natural_landmark',
 'park',
 'religious_site',
 'scenic',
 'urban',
 'wildlife']

image_size = (150, 150)

@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload an Image</title>
    </head>
    <body>
        <h1>Upload an Image for Prediction</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <label for="file">Choose an image:</label>
            <input type="file" name="file" id="file" accept="image/*" required>
            <br><br>
            <button type="submit">Upload and Predict</button>
        </form>
    </body>
    </html>
    """


def preprocess_and_predict(img, model, target_size=image_size, top_n=3):
    """
    Preprocess the input image and make a prediction with the model.
    """
    # Convert PIL image to target size and preprocess
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image

    # Make prediction
    predictions = model.predict(img_array)

    # Get top_n predictions
    top_n_indices = np.argsort(predictions[0])[::-1][:top_n]
    top_classes = [(class_labels[i], float(predictions[0][i])) for i in top_n_indices]

    return top_classes


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image file from the request
        file = request.files.get("file")

        if not file or file.filename == "":
            return jsonify({"status": False, "code": 400, "message": "No file selected", "data": None}), 400

        # Open image file
        img = Image.open(file)

        # Process and predict
        top_predictions = preprocess_and_predict(img, model)

        # Generate recommendations for the top predicted genres
        recommendations = []

        for genre, count in zip([top_predictions[0][0], top_predictions[1][0], top_predictions[2][0]], [4, 3, 3]):
            genre_recommendations = recommendation_data[recommendation_data["genre"].str.lower() == genre.lower()]

            if not genre_recommendations.empty:
                sampled_recommendations = genre_recommendations.sample(min(count, len(genre_recommendations))).to_dict(orient="records")
                recommendations.extend(sampled_recommendations)

        # Format response
        return jsonify({
            "predicted_genres": top_predictions,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"status": False, "code": 500, "message": str(e), "data": None}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0")
