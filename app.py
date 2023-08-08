#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib

# Load the model
model = joblib.load("best_model_svc.pkl")

# Load the multilabel binarizer
multilabel_binarizer = joblib.load("binarizer.pkl")

# Load the TfidfVectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    text_input = request.form["text_input"]

    # Convert the text into a numerical feature vector using your pre-processing steps
    features = pre_process_text(text_input, vectorizer)

    # Make prediction
    prediction = model.predict(features)

    # Inverse multilabel binarizer
    tags_predict = multilabel_binarizer.inverse_transform(prediction)

    return render_template("index1.html", prediction_text="The predicted tags are {}".format(tags_predict))

def pre_process_text(text, vectorizer):
    # Replace this function with your own pre-processing code to convert the text input into numerical features

    from texthero import preprocessing
    import texthero as hero

    dat = pd.DataFrame([text])

    custom_pipeline = [preprocessing.fillna,
                       preprocessing.lowercase,
                       preprocessing.remove_digits,
                       preprocessing.remove_punctuation,
                       preprocessing.remove_diacritics,
                       preprocessing.remove_stopwords,
                       preprocessing.remove_whitespace
                      ]

    # Pass the custom_pipeline to the pipeline argument
    cleantext = hero.clean(dat[0], pipeline=custom_pipeline)

    # Transform the preprocessed text using the loaded vectorizer
    X = vectorizer.transform(cleantext)
    return X.toarray()

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Get the port from the environment variable or use 5000 by default
    app.run(host='0.0.0.0', port=port)



# In[ ]:




