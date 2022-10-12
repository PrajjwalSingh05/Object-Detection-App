from asyncore import write
from distutils.command.clean import clean
import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from PIL import Image
import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Object Detection App",
    layout='wide',
    page_icon="🛃"
)
def introduction():
    # st.image('vg.gif', width=None)
    st.markdown("""
        
    Name : Prajjwal Singh
    \nQualification : Bachelor ofComputer Application(BCA)
    \nStream : Computer Science
    \nUniversity : University of Lucknow
    \nLocation : Lucknow, INDIA
    \nThis Project Perfrom Prdiction Wheather You Have Donated Blood Or not 
        
        - The Libraries I used in Project are:
            Matplotlib Explore here
            Sklearn Explore Here
            Streamlit Explore here
            Pandas 
            imbelearn 
        - Their Following Tasks are Implemented in the Project:
            Data Preparation and Cleaning
            Model design
            Best Feature Selectio 
            References and Future Work
    """)
    


 
def load_model():
    try:
        model = tf.keras.models.load_model('cifar_model.h5')
        return model
    except Exception as e:
        st.error(e)
def predict_class(model, image, shape=(32, 32)):
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    image = np.expand_dims(image, axis=0)
    # st.write(image.shape)
    image = tf.image.resize(image, shape)
    output = model.predict(image)
    return class_names[output.argmax()]
with st.spinner("Loading Model Into Memory..."):
    model = load_model()
def execute():
    st.write(" ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']")
    # with st.spinner("Loading Model Into Memory..."):
    #     model= load_model()

    st.title("Object Detection App")
    st.write("This is a simple object detection web app to detect objects in images.")

    with st.form("Form1"):
        file = st.file_uploader("Upload an image", type=["jpg", "png"])
        submit = st.form_submit_button("Submit")

    c1 , c2 = st.columns(2)
    if file:
        c1.image(file)
        image = Image.open(file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        result = predict_class(model, image)
        c2.markdown(f'''# Prediction
        {result}
    ''')



# image=Image.open("house_sale.jpg")

st.header("House Price Prediction ")

options = ['Project Introduction', 'Execution']

sidebar = st.sidebar

sidebar.title('User Options')

selOption = sidebar.selectbox("Select an Option", options)

if selOption == options[0]:
    introduction()
elif selOption == options[1]:
    execute()
        