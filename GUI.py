import keras
import os
import tensorflow as tf
from keras.models import Sequential,load_model,model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import numpy as np
import json
import cv2
import time

guiHeight = 400
guiWidth = 500
picHeight = 256
picWidth = 256
infoPadding = 35
defaultW = 224
defaultH = 224
img=None
backgroundColor="#505455"
fontColor="white"

class Lotfi(tk.Entry):
    def __init__(self, master=None, **kwargs):
        self.var = tk.StringVar()
        tk.Entry.__init__(self, master, textvariable=self.var, **kwargs)
        self.old_value = ''
        self.var.trace('w', self.check)
        self.get, self.set = self.var.get, self.var.set
    def check(self, *args):
        if self.get().isdigit():
            # the current value is only digits; allow this
            self.old_value = self.get()
        else:
            # there's non-digit characters in the input; reject this
            self.set(self.old_value)

def getPrediction():
    if img is None:
        messagebox.showinfo("Error","You must select an image first.")
    else:
        try:
            infoLabel.config(text="Loading...")
            master.update()
            json_file = open("model.json","r")
        except:
            messagebox.showinfo("Error","model.json wasn't found.")
            infoLabel.config(text="Move model.json to the application directory.")
            return
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        try:
            model.load_weights("Weights/model_weights.h5")
        except:
            messagebox.showinfo("Error","model_weights.h5 wasn't found.")
            infoLabel.config(text="Move model_weights.h5 to the application directory.")
            return
        try:
            json_file = open("class_names.json","r")
        except:
            messagebox.showinfo("Error","class_names.json wasn't found.")
            infoLabel.config(text="Move class_names.json to the application directory.")
            return
        class_names = json.loads(json_file.read())
        json_file.close()
        class_names = dict([[v,k] for k,v in class_names.items()])
        inputWidth = int(wEntry.get())
        inputHeight = int(hEntry.get())
        imgPred = img.resize((inputWidth,inputHeight))
        imgPred = np.array(imgPred)
        imgPred = np.expand_dims(imgPred, axis=0)
        try:
            predictions = model.predict(imgPred)
            print(predictions)
            y_pred = [np.argmax(probas) for probas in predictions]
            # messagebox.showinfo("Prediction",class_names[y_pred[0]])
            infoLabel.config(text="Predicted: "+class_names[y_pred[0]])
        except:
            messagebox.showinfo("Error","Image size does not match model input.")

def loadImg():
    global img
    Tk().withdraw()
    imgName = Path(askopenfilename(title = "Select file",filetypes = [("Picture files",("*.jpg","*.png","*.gif","*.ppm","*.ico"))]))
    if str(imgName) is not ".":
        try:
            img = Image.open(imgName)
            img = img.convert("RGB")
        except:
            messagebox.showinfo("Error","The image you tried to open was corrupted.")
            return
        displayImage()

def displayImage():
    global imgDisp
    imgDisp = img.resize((picWidth,picHeight))
    imgDisp = ImageTk.PhotoImage(imgDisp)
    can.delete("all")
    can.create_image((guiWidth-picWidth)/2,(guiHeight-picHeight)/2-infoPadding,image=imgDisp,anchor=NW)
    can.create_rectangle((guiWidth-picWidth)/2-1,(guiHeight-picHeight)/2-1-infoPadding,guiWidth-((guiWidth-picWidth)/2),guiHeight-((guiHeight-picHeight)/2)-infoPadding)
    infoLabel.config(text="Press get prediction.")

def takePic():
    global img
    cam = cv2.VideoCapture(0)
    frame = None
    while True:
        ret, frame = cam.read()
        try:
            cv2.imshow("Press space to take a picture", frame)
        except:
            messagebox.showinfo("Error","You don't have a camera or your camera is disabled.")
            return
        if not ret:
            break
        k = cv2.waitKey(1)
        if k%256 == 32:
            break
        if cv2.getWindowProperty("Press space to take a picture",1) == -1:
            cam.release()
            cv2.destroyAllWindows()
            return
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    cam.release()
    cv2.destroyAllWindows()
    displayImage()

master = Tk()
master.configure(bg=backgroundColor)
frame = Frame(master,bg=backgroundColor)
frame.pack(side=RIGHT,fill=Y)
scrollbar = Scrollbar(frame, orient=VERTICAL)
listbox = Listbox(frame, yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)
listLabel = Label(frame,text="List of classes",bg=backgroundColor,fg=fontColor)
listLabel.pack(side=TOP)
scrollbar.pack(side=RIGHT, fill=Y)
listbox.pack(side=LEFT, fill=BOTH, expand=1)
try:
    json_file = open("class_names.json","r")
    class_names = json.loads(json_file.read())
    json_file.close()
    for x in class_names.keys():
        listbox.insert(END,x)
except:
    messagebox.showinfo("Error","class_names.json wasn't found.")
infoLabel = Label(text="Select an image.",bg=backgroundColor,fg=fontColor)
infoLabel.pack(side=TOP,pady=(infoPadding,0))
frameH = Frame(master)
frameW = Frame(master,bg=backgroundColor)
frameButtons = Frame(master)
frameButtons.pack(side=BOTTOM)
frameW.pack(side=BOTTOM)
frameH.pack(side=BOTTOM)
master.title("Image Identification")
can = Canvas(master,width=guiWidth,height=guiHeight,bg=backgroundColor,highlightthickness=0)
can.pack()
can.create_rectangle((guiWidth-picWidth)/2,(guiHeight-picHeight)/2-infoPadding,guiWidth-((guiWidth-picWidth)/2),guiHeight-((guiHeight-picHeight)/2)-infoPadding,fill="grey")
hLabel = Label(frameH,text="Input height:",bg=backgroundColor,fg=fontColor)
hEntry = Lotfi(frameH)
wLabel = Label(frameW,text="Input width:",bg=backgroundColor,fg=fontColor)
wEntry = Lotfi(frameW)
hLabel.pack(side=LEFT)
hEntry.pack(side=LEFT)
wLabel.pack(side=LEFT,padx=(5,0))
wEntry.pack(side=LEFT)
hEntry.delete(0, END)
hEntry.insert(0, defaultH)
wEntry.delete(0, END)
wEntry.insert(0, defaultW)
button1=tk.Button(frameButtons,text="Select a picture",command=loadImg,bg=backgroundColor,fg=fontColor)
button2=tk.Button(frameButtons,text="Take a picture",command=takePic,bg=backgroundColor,fg=fontColor)
button3=tk.Button(frameButtons,text="Get prediction",command=getPrediction,bg=backgroundColor,fg=fontColor)
button4=tk.Button(frameButtons,text="Quit",command=master.quit,bg=backgroundColor,fg=fontColor)
button1.pack(side=LEFT)
button2.pack(side=LEFT)
button3.pack(side=LEFT)
button4.pack(side=LEFT)
master.protocol("WM_DELETE_WINDOW",master.quit)
mainloop()