from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
import pickle

def extract_features(directory):
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[-2].output)

    features = {}
    for img_name in os.listdir(directory):
        filename = directory + "/" + img_name
        image = load_img(filename, target_size=(299, 299))
        image = img_to_array(image)
        image = image.reshape((1, 299, 299, 3))
        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)
        features[img_name] = feature

    with open("features/features.pkl", "wb") as f:
        pickle.dump(features, f)