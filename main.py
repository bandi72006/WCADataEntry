from tkinter import *   
from tkinter.ttk import *
from cameraDetection import *

root = Tk()             

root.geometry("480x480")

btn = Button(root, text = "Input score card",
                          command = runCamera)

btn.pack(side = 'top')   
 
root.mainloop()