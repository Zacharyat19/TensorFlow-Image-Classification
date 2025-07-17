import os
import json
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import webbrowser
from pathlib import Path
from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from pandas import DataFrame
from win32api import GetMonitorInfo, MonitorFromPoint
from keras.models import model_from_json

# GUI Configuration
GUI_HEIGHT, GUI_WIDTH = 400, 500
PIC_HEIGHT, PIC_WIDTH = 256, 256
INFO_PADDING = 35
DEFAULT_W, DEFAULT_H = 64, 64
BACKGROUND_COLOR = "#505455"
FONT_COLOR = "white"
PIC_WIN_TEXT = "Take a picture"

# Globals
chart = img = loaded_model_json = weightsDone = class_names = None

# Helper to open URLs
def callback(url):
    webbrowser.open_new(url)

# Entry widget for only numeric input
class Lotfi(tk.Entry):
    def __init__(self, master=None, **kwargs):
        self.var = tk.StringVar()
        super().__init__(master, textvariable=self.var, **kwargs)
        self.old_value = ''
        self.var.trace('w', self.check)
        self.get, self.set = self.var.get, self.var.set

    def check(self, *args):
        if self.get().isdigit():
            self.old_value = self.get()
        else:
            self.set(self.old_value)

# Handle model prediction
def getPrediction():
    global chart, loaded_model_json, weightsDone, class_names, model
    if img is None:
        messagebox.showinfo("Error", "You must select an image first.")
        return

    try:
        if loaded_model_json is None:
            infoLabel.config(text="Loading...")
            master.update()
            with open("model.json", "r") as f:
                loaded_model_json = f.read()
            model = model_from_json(loaded_model_json)

        if weightsDone is None:
            model.load_weights("model_weights.h5")
            weightsDone = True

        if class_names is None:
            with open("class_names.json", "r") as f:
                class_names = json.loads(f.read())
            class_names = {v: k for k, v in class_names.items()}

        inputWidth, inputHeight = int(wEntry.get()), int(hEntry.get())
        imgPred = img.resize((inputWidth, inputHeight))
        imgPred = np.expand_dims(np.array(imgPred), axis=0)
        predictions = model.predict(imgPred)

    except Exception as e:
        messagebox.showinfo("Error", str(e))
        infoLabel.config(text="Ensure model files and sizes match.")
        return

    if chart:
        chart.pack_forget()

    y_pred = [np.argmax(probas) for probas in predictions]
    infoLabel.config(text="Predicted: " + class_names[y_pred[0]])

    predictions = predictions[0]
    top_idx = np.argsort(predictions)[-3:][::-1]
    data = {"Class": [class_names[x] for x in top_idx], "percentage": [predictions[x] for x in top_idx]}
    dataFr = DataFrame(data).groupby('Class').sum()

    figure = plt.Figure(figsize=(GUI_WIDTH/75, 450/75), dpi=75)
    ax = figure.add_subplot()
    chart_type = FigureCanvasTkAgg(figure, frameGraph)
    chart = chart_type.get_tk_widget()
    chart.pack(side=BOTTOM)
    dataFr.plot(kind='bar', ax=ax, rot=0)

# Load an image from file
def loadImg():
    global img
    Tk().withdraw()
    imgName = Path(askopenfilename(title="Select file", filetypes=[("Picture files", ("*.jpg", "*.png", "*.gif", "*.ppm", "*.ico"))]))
    if imgName != Path('.'):
        try:
            img = Image.open(imgName).convert("RGB")
            displayImage()
        except:
            messagebox.showinfo("Error", "The image you tried to open was corrupted.")

# Display the selected image in the GUI
def displayImage():
    global imgDisp
    imgDisp = img.resize((PIC_WIDTH, PIC_HEIGHT))
    imgDisp = ImageTk.PhotoImage(imgDisp)
    can.delete("all")
    can.create_image((GUI_WIDTH - PIC_WIDTH) / 2, (GUI_HEIGHT - PIC_HEIGHT) / 2 - INFO_PADDING, image=imgDisp, anchor=NW)
    can.create_rectangle((GUI_WIDTH - PIC_WIDTH) / 2 - 1, (GUI_HEIGHT - PIC_HEIGHT) / 2 - 1 - INFO_PADDING,
                         GUI_WIDTH - (GUI_WIDTH - PIC_WIDTH) / 2, GUI_HEIGHT - (GUI_HEIGHT - PIC_HEIGHT) / 2 - INFO_PADDING)
    infoLabel.config(text="Press get prediction.")

# Use the webcam to take a picture
def takePic():
    global img
    cam = cv2.VideoCapture(0)
    cv2.namedWindow(PIC_WIN_TEXT, cv2.WINDOW_NORMAL)
    cv2.moveWindow(PIC_WIN_TEXT, GUI_WIDTH, 0)

    tWinH = master.winfo_height()
    _, _, cWinW, cWinH = cv2.getWindowImageRect(PIC_WIN_TEXT)
    ratio = tWinH / cWinH
    cv2.resizeWindow(PIC_WIN_TEXT, int(ratio * cWinW), int(ratio * cWinH))

    infoLabel.config(text="Press space to take a picture.")
    master.update()

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.imshow(PIC_WIN_TEXT, frame)
        k = cv2.waitKey(1)
        if k % 256 == 32:
            break
        if cv2.getWindowProperty(PIC_WIN_TEXT, 1) == -1:
            cam.release()
            cv2.destroyAllWindows()
            return

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cam.release()
    cv2.destroyAllWindows()
    displayImage()

# GUI Setup
master = Tk()
master.geometry("+-8+0")
monitor_info = GetMonitorInfo(MonitorFromPoint((0, 0)))
work_area = monitor_info.get("Work")
master.maxsize(work_area[2], work_area[3] - 35)
master.configure(bg=BACKGROUND_COLOR)

frame = Frame(master, bg=BACKGROUND_COLOR)
frame.pack(side=RIGHT, fill=Y)
scrollbar = Scrollbar(frame, orient=VERTICAL)
listbox = Listbox(frame, yscrollcommand=scrollbar.set)
listLabel = Label(frame, text="List of classes", bg=BACKGROUND_COLOR, fg=FONT_COLOR)
listLabel.pack(side=TOP)
scrollbar.pack(side=RIGHT, fill=Y)
listbox.pack(side=LEFT, fill=BOTH, expand=1)
scrollbar.config(command=listbox.yview)

try:
    with open("class_names.json", "r") as f:
        class_names = json.loads(f.read())
        for name in class_names:
            listbox.insert(END, name)
        class_names = {v: k for k, v in class_names.items()}
except:
    messagebox.showinfo("Error", "class_names.json wasn't found.")

infoLabel = Label(text="Select an image.", bg=BACKGROUND_COLOR, fg=FONT_COLOR, font=("Helvetica", 15))
infoLabel.pack(side=TOP, pady=(INFO_PADDING, 0))
link = Label(text="Web version", bg=BACKGROUND_COLOR, fg="#00f9ff", cursor="hand2")
link.pack(side=TOP)
link.bind("<Button-1>", lambda e: callback("http://nwapw-tf.com/"))

# Frames
frameH = Frame(master)
frameW = Frame(master, bg=BACKGROUND_COLOR)
frameButtons = Frame(master)
frameGraph = Frame(master)
frameGraph.pack(side=BOTTOM)
frameButtons.pack(side=BOTTOM)
frameW.pack(side=BOTTOM)
frameH.pack(side=BOTTOM)

master.title("Image Identification")
can = Canvas(master, width=GUI_WIDTH, height=GUI_HEIGHT, bg=BACKGROUND_COLOR, highlightthickness=0)
can.pack()
can.create_rectangle((GUI_WIDTH - PIC_WIDTH)/2, (GUI_HEIGHT - PIC_HEIGHT)/2 - INFO_PADDING,
                     GUI_WIDTH - (GUI_WIDTH - PIC_WIDTH)/2, GUI_HEIGHT - (GUI_HEIGHT - PIC_HEIGHT)/2 - INFO_PADDING,
                     fill="grey")

# Height/Width input
hLabel = Label(frameH, text="Input height:", bg=BACKGROUND_COLOR, fg=FONT_COLOR)
hEntry = Lotfi(frameH)
wLabel = Label(frameW, text="Input width:", bg=BACKGROUND_COLOR, fg=FONT_COLOR)
wEntry = Lotfi(frameW)

hLabel.pack(side=LEFT)
hEntry.pack(side=LEFT)
wLabel.pack(side=LEFT, padx=(5, 0))
wEntry.pack(side=LEFT)

hEntry.delete(0, END)
hEntry.insert(0, DEFAULT_H)
wEntry.delete(0, END)
wEntry.insert(0, DEFAULT_W)

# Buttons
button1 = Button(frameButtons, text="Select a picture", command=loadImg, bg=BACKGROUND_COLOR, fg=FONT_COLOR)
button2 = Button(frameButtons, text="Take a picture", command=takePic, bg=BACKGROUND_COLOR, fg=FONT_COLOR)
button3 = Button(frameButtons, text="Get prediction", command=getPrediction, bg=BACKGROUND_COLOR, fg=FONT_COLOR)
button4 = Button(frameButtons, text="Quit", command=master.quit, bg=BACKGROUND_COLOR, fg=FONT_COLOR)

button1.pack(side=LEFT)
button2.pack(side=LEFT)
button3.pack(side=LEFT)
button4.pack(side=LEFT)

master.protocol("WM_DELETE_WINDOW", master.quit)
mainloop()
