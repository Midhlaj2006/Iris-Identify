#imports
from pathlib import WindowsPath
import streamlit as st
import pandas as pd
import pickle

#open model
pkl =open("IrisDetect.pkl", "rb")
model =pickle.load(pkl)

#title
st.title("Plant Recognizer")
st.markdown("Provide us the details in **Centimetre**")

#image
from PIL import Image
png = Image.open('img.png')
st.image(png, width=300)

#inputs
SepalLength =st.text_input("Sepal Length")
SepalWidth =st.text_input("Sepal Width")
PetalLength =st.text_input("Petal Length")
PetalWidth =st.text_input("Petal Width")
inps =[SepalLength,SepalWidth,PetalLength,PetalWidth]

#buttons
result =st.button("See result")

if result:#If button got pressed:   
    try:
        data=[float(num) for num in (inps)]
        pred = model.predict([data])
        
        #the freaking RESULTTTTT
        st.title(f"This can be an {pred.all()}")
        #why am I adding balllons?
        #you might not think I don't know how to
        st.balloons()
        
    except ValueError:
        st.markdown("please fill all text boxes!")

#expander
table =st.beta_expander(label="Dataset")
with table:#If you need to know the pre defined values
    st.table(pd.read_csv("Iris.csv").drop(columns="Id"))

# This code is in streamlit, so run it with
#    streamlit run main.py  