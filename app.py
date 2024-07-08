import gradio as gr 
import tensorflow as tf 
import numpy as np 

modelo = tf.keras.models.load_model("mnist_model.h5")

def clasificar_imagenes(img):
    img = np.reshape(img, (1, 28, 28, 1)).astype("float32") / 255
    predicciones = modelo.predict(img)
    digito_predicho = np.argmax(predicciones)
    return str(digito_predicho)

interfaz = gr.Interface(fn=clasificar_imagenes , inputs="sketchpad", outputs="label")
interfaz.launch()