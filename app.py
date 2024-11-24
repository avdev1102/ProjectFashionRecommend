import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st
import random
import re
import cv2

# Load dataset
csv_file_path = './styles_validated.csv'
images_path = './images'
df = pd.read_csv(csv_file_path)

# Streamlit App
st.title("Fashion Outfit Generator")

# Initialize session state
if "selected_outfit" not in st.session_state:
    st.session_state["selected_outfit"] = None

if "outfits" not in st.session_state:
    st.session_state["outfits"] = []

# Landing Screen: Gender and Usage Selection
gender = st.selectbox("Select Gender", df['gender'].unique())
usage = st.selectbox("Select Usage", df['usage'].unique())

# Filter Data Based on User Selection
filtered_df = df[(df['gender'] == gender) & (df['usage'] == usage)]

# Display color selection for each category
st.header("Select Colors for Each Category")
topwear_color = st.selectbox("Select Topwear Color", filtered_df['baseColour'].unique())
bottomwear_color = st.selectbox("Select Bottomwear Color", filtered_df['baseColour'].unique())
shoes_color = st.selectbox("Select Shoes Color", filtered_df['baseColour'].unique())

Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))
file_names = [re.search(r'images\\(\d+)\.jpg', img).group(1) for img in filenames]

def extract_features_with_color(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    resnet_features = model.predict(img_preprocess).flatten()
    resnet_features = resnet_features / norm(resnet_features)

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([img_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    color_features = np.resize(hist, resnet_features.shape)
    return resnet_features + color_features

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Generate Outfits Button
if st.button("Generate Outfits"):
    outfits = []
    for _ in range(3):
        topwear = filtered_df[(filtered_df['subCategory'] == 'Topwear') & 
                              (filtered_df['baseColour'] == topwear_color)].sample(1)
        bottomwear = filtered_df[(filtered_df['subCategory'] == 'Bottomwear') & 
                                 (filtered_df['baseColour'] == bottomwear_color)].sample(1)
        shoes = filtered_df[(filtered_df['subCategory'] == 'Shoes') & 
                            (filtered_df['baseColour'] == shoes_color)].sample(1)
        outfit = pd.concat([topwear, bottomwear, shoes], ignore_index=True)
        outfits.append(outfit)

    st.session_state["outfits"] = outfits

# Display outfits and allow user selection
if st.session_state["outfits"]:
    st.write("Generated Outfits:")
    outfit_selection = st.radio(
        "Select an Outfit",
        options=[f"Outfit {i + 1}" for i in range(len(st.session_state["outfits"]))],
        index=0,
        key="outfit_selection"
    )

    selected_outfit_index = int(outfit_selection.split()[-1]) - 1
    st.session_state["selected_outfit"] = st.session_state["outfits"][selected_outfit_index]

    st.write("Selected Outfit:")
    for _, item in st.session_state["selected_outfit"].iterrows():
        img_path = os.path.join(images_path, f"{item['id']}.jpg")
        st.image(img_path, caption=f"{item['productDisplayName']} ({item['baseColour']})", width=100)

# Modify the selected outfit
if st.session_state["selected_outfit"] is not None:
    st.write("Modify Your Outfit:")
    modify_choice = st.selectbox(
        "What would you like to modify?",
        ["None", "Topwear", "Bottomwear", "Shoes"],
        key="modify_choice"
    )

    if modify_choice != "None":
        color_options = filtered_df[filtered_df['subCategory'] == modify_choice]['baseColour'].unique()
        new_color = st.selectbox(f"Select new {modify_choice} color", color_options, key="new_color")

        modified_item = filtered_df[
            (filtered_df['subCategory'] == modify_choice) &
            (filtered_df['baseColour'] == new_color)
        ].sample(1)

        item_index = {"Topwear": 0, "Bottomwear": 1, "Shoes": 2}[modify_choice]
        st.session_state["selected_outfit"].iloc[item_index] = modified_item.iloc[0]

        st.write("Updated Outfit:")
        for _, item in st.session_state["selected_outfit"].iterrows():
            img_path = os.path.join(images_path, f"{item['id']}.jpg")
            st.image(img_path, caption=f"{item['productDisplayName']} ({item['baseColour']})", width=100)

# Confirm Final Outfit
if st.session_state["selected_outfit"] is not None:
    if st.button("Confirm Outfit"):
        st.write("Your final outfit is confirmed!")
        for _, item in st.session_state["selected_outfit"].iterrows():
            img_path = os.path.join(images_path, f"{item['id']}.jpg")
            st.image(img_path, caption=f"{item['productDisplayName']} ({item['baseColour']})", width=100)
