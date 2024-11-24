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

Image_features = pkl.load(open('Images_features.pkl','rb'))
filenames = pkl.load(open('filenames.pkl','rb'))
file_names = [re.search(r'images\\(\d+)\.jpg', img).group(1) for img in filenames]

def extract_features_with_color(image_path, model):
    # Extract ResNet50 features
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    resnet_features = model.predict(img_preprocess).flatten()
    resnet_features = resnet_features / norm(resnet_features)

    # Extract color histogram features
    img_bgr = cv2.imread(image_path)  # OpenCV loads images in BGR format
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([img_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])  # 8 bins per channel
    hist = cv2.normalize(hist, hist).flatten()

    color_features = np.resize(hist, resnet_features.shape)
    return resnet_features + color_features

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tf.keras.models.Sequential([model,GlobalMaxPool2D()])


# Generate Outfits Button
if st.button("Generate Outfits"):
    filtered = []
    # Filter Data Based on Color Preferences
    topwear_filtered = filtered_df[(filtered_df['subCategory'] == 'Topwear') & 
                                   (filtered_df['baseColour'] == topwear_color)]
    bottomwear_filtered = filtered_df[(filtered_df['subCategory'] == 'Bottomwear') & 
                                      (filtered_df['baseColour'] == bottomwear_color)]
    shoes_filtered = filtered_df[(filtered_df['subCategory'] == 'Shoes') & 
                                 (filtered_df['baseColour'] == shoes_color)]
    
    filtered.extend([topwear_filtered, bottomwear_filtered, shoes_filtered])

    # Ensure there are enough items to generate outfits
    if len(topwear_filtered) < 3 or len(bottomwear_filtered) < 3 or len(shoes_filtered) < 3:
        st.error("Not enough items to generate outfits. Try different colors or options.")
    else:
        # Select Random Items for First Outfit
        topwear = topwear_filtered.sample(1)
        bottomwear = bottomwear_filtered.sample(1)
        shoes = shoes_filtered.sample(1)

        first_outfit = pd.concat([topwear, bottomwear, shoes], ignore_index=True)

        outfits = []
        outfits.append(first_outfit)
        
        filenames_each = []
        
        for ele in filtered:
            filenames_each.append(ele['id'].astype(str).tolist())

        similar_outfits = []
        for ind, item in first_outfit.iterrows():
            input_img_features = extract_features_with_color(os.path.join(images_path, f"{item['id']}.jpg"), model)
            match_indices = [i for i, element in enumerate(file_names) if element in filenames_each[ind]]
            features_matches = [Image_features[ind] for ind in match_indices]
            neighbors = NearestNeighbors(n_neighbors=3, algorithm='brute', metric='euclidean')
            neighbors.fit(features_matches)
            distance, indices = neighbors.kneighbors([input_img_features])
            similar_outfits.append(indices[0][1:])

        for j in range(len(similar_outfits[0])):
            outfit = []
            for i in range(len(similar_outfits)):
                outfit.append(similar_outfits[i][j]) 

            outfit_concat = []
            for i, ind in enumerate(outfit):
                im = df[df['id'].astype(str) == filenames_each[i][ind]]
                outfit_concat.append(im)

            outfits.append(pd.concat(outfit_concat))
            #st.image(os.path.join(images_path, f"{filenames_each[i][ind]}.jpg"), caption=f"{im.iloc[0]["productDisplayName"]} ({im.iloc[0]['baseColour']})", width=80)
        
        print(outfits, "/n")

        columns = st.columns(3)
        # Iterate over the columns
        for i, col in enumerate(columns):
            col.write(f"Outfit {i+1}")
            for _, item in outfits[i].iterrows():
                img_path = os.path.join(images_path, f"{item['id']}.jpg")
                col.image(img_path, caption=f"{item['productDisplayName']} ({item['baseColour']})", width=80)


