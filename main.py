# ---------------------------------------
# project run command
# streamlit run app.py
# #change app name with your settled name
# ---------------------------------------
import os
import streamlit as st
import tensorflow as tf
import numpy as np
#-------------------------
st.header('Texture Classification Ml Model')
st.subheader('Welcome')
densenet201_model = tf.keras.models.load_model('my_model.h5')
class_names = ["Extra G1", "Extra G2","Extra G3", "Extra M100","Extra M150", "Extra M200","Extra M250", "Terno 30","Terno 50", "Terno 90","Tm 10","Tm 40","Tm 50","Tm 70"]
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(224,224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = densenet201_model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    #----------new----testing-------
    predicted_class_index = np.argmax(result)
    predicted_class_name = class_names[predicted_class_index]
    #-----------------------
    # Get the image file name
    image_name = os.path.basename(image_path)
    outcome = f'Results:  \n\n  Uploaded Image name: {image_name}  \n\n  Predicted to class of: {predicted_class_name}.'

    return outcome, image_name

uploaded_file = st.file_uploader('Upload Texture Image')
if uploaded_file is not None:
    os.makedirs(os.path.join('upload'), exist_ok=True)
    with open(os.path.join('upload',uploaded_file.name),'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, width=224)
    outcome, image_name = classify_images(os.path.join('upload', uploaded_file.name))
    st.markdown(outcome)
    st.write(f"Predicted image name/color code: {image_name}")
