import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

model = tensorflow.keras.models.load_model("pokedex_mobilenetv2-2.keras")
classlabel = ["Aerodactyl","Bulbasaur","Charmander","Dratini","Fearow","Mewtwo","Pikachu","Psyduck","Spearow","Squirtle"]

def image_processing(path):
    img = image.load_img(path, target_size=(227,227))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr/225.0

    return img_arr

def predict(path):
    img_arr = image_processing(path)
    predictions = model.predict(img_arr)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return classlabel[predicted_class]

# predicted_class = predict(model_, "pixel-3316924_1280.jpg", classlabel)
# print(predicted_class)