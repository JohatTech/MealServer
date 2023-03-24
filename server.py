from typing import Union
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
import io 
from PIL import Image
import numpy as np 
import pandas as pd 
import tensorflow as tf

app =FastAPI()

model = tf.keras.models.load_model('model')

@app.post("/classify/")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    
    # Load the image using Pillow
    img = Image.open(io.BytesIO(contents))
    
    # Resize the image to match the input shape of the model
    img = img.resize((192, 192))
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Add an extra dimension to the array to represent the batch size of 1
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the pixel values to be between 0 and 1
    img_array = img_array / 255.0
    
    # Make a prediction using the loaded model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    food_classes = list(pd.read_csv('aiy_food_V1_labelmap (1).csv')["name"])

    return {"prediction class": food_classes[predicted_class]}
