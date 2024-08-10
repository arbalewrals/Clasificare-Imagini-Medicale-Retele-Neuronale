import tensorflow as tf
import cv2
import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk
import numpy as np

my_w = tk.Tk()
my_w.geometry("500x400")  # Size of the window
my_w.title('Tumor recognition')
my_font1=('times', 18, 'bold')
l1 = tk.Label(my_w,text='Add Photo',width=35,font=my_font1)
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Upload File',width=20,command = lambda:upload_file())
b1.grid(row=2,column=1)
b2 = tk.Button(my_w, text='Predict',width=20,command = lambda:predict())
b2.grid(row=4,column=1)
l2 = tk.Label(my_w,text='Predicted',width=35,font=my_font1)
l2.grid(row=5,column=1)

loaded_model = tf.keras.models.load_model("model.h5")

def predict():
    predicted = np.argmax(loaded_model.predict(np.expand_dims(resize, 0)), axis=1)
    if predicted==0:
        l2.configure(text='Predicted: No Tumor')
    else:
        l2.configure(text='Predicted: Tumor')


def upload_file():
    global img
    f_types = [('Jpg Files', '*.jpg'),('Jpeg Files','*.jpeg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    nume = str(os.path.basename(filename))
    img_tf=cv2.imread(f'testing/{nume}')
    global resize
    resize = tf.image.resize(img_tf, (244, 244))
    img = Image.open(filename)
    img_resized=img.resize((244,244))
    img = ImageTk.PhotoImage(img_resized)
    b2 =tk.Button(my_w,image=img)
    b2.grid(row=3,column=1)

my_w.mainloop()
