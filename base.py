import tensorflow as tf
import cv2
import imghdr
import os
import matplotlib.pyplot as plt 
import numpy as np
import streamlit as st
from PIL import Image,ImageOps
from tensorflow.keras.models import load_model


def classifier(image_data, model):
    size=(256,256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img_reshape = np.expand_dims(image/255,0)

    prediction = model.predict(img_reshape)


    return prediction








        


















