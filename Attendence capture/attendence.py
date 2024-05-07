import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import simpledialog
import time
from tkinter import messagebox
import os
from keras.utils.np_utils import to_categorical
26
import numpy as np
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from tkinter import *
import random
from datetime import date
class App:
 global classifier
 global labels
 global X_train
 global Y_train
 global text
 global img_canvas
 global cascPath
 global faceCascade
 global tf1, tf2
 global capture_img
 global student_details
 def __init__(self, window, window_title, video_source=0):
27
 global cart
 global text
 cart = []
 self.window = window
 self.window.title("Automatic Attendance Management System using Face Detection")
 self.window.geometry("1300x1200")
 self.video_source = video_source
 self.vid = MyVideoCapture(self.video_source)
 self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
 self.canvas.pack()
 self.font1 = ('times', 13, 'bold')
 self.capture_img = None
 font = ('times', 16, 'bold')
 title = Label(window, text='Automatic Attendance Management System using Face 
Detection')
 title.config(bg='darkviolet', fg='gold') 
 title.config(font=font) 
 title.config(height=3, width=120) 
 title.place(x=0,y=5) 
 self.l1 = Label(window, text='STUDENT ID')
 self.l1.config(font=self.font1)
 self.l1.place(x=50,y=500)
28
 self.tf1 = Entry(window,width=30)
 self.tf1.config(font=self.font1)
 self.tf1.place(x=50,y=550)
 self.l2 = Label(window, text='STUDENT NAME')
 self.l2.config(font=self.font1)
 self.l2.place(x=350,y=500)
 self.tf2 = Entry(window,width=60)
 self.tf2.config(font=self.font1)
 self.tf2.place(x=350,y=550)
 
 self.btn_snapshot=tkinter.Button(window, text="Capture Face Image", 
command=self.capturePerson)
 self.btn_snapshot.place(x=50,y=600)
 self.btn_snapshot.config(font=self.font1)
 
 self.btn_train=tkinter.Button(window, text="Train Model", command=self.trainmodel)
 self.btn_train.place(x=250,y=600)
 self.btn_train.config(font=self.font1)
 
 self.btn_predict=tkinter.Button(window, text="Take Attendance", command=self.predict)
29
 self.btn_predict.place(x=400,y=600)
 self.btn_predict.config(font=self.font1)
 '''
 self.img_canvas = tkinter.Canvas(window, width = 200, height = 200)
 self.img_canvas.place(x=10,y=250)
 '''
 self.text=Text(window,height=35,width=65)
 scroll=Scrollbar(self.text)
 self.text.configure(yscrollcommand=scroll.set)
 self.text.place(x=1000,y=90)
 self.text.config(font=self.font1)
 self.cascPath = "haarcascade_frontalface_default.xml"
 self.faceCascade = cv2.CascadeClassifier(self.cascPath)
 self.student_details = []
 if os.path.exists('details.txt'):
 with open("details.txt", "r") as file:
 for line in file:
 line = line.strip('\n')
 line = line.strip()
 self.student_details.append(line)
 self.delay = 15
30
 self.update()
 self.window.config(bg='turquoise')
 self.window.mainloop()
 def getID(self,name):
 index = 0
 for i in range(len(labels)):
 if labels[i] == name:
 index = i
 break
 return index
 def capturePerson(self):
 global capture_img
 option = 0
 ret, frame = self.vid.get_frame()
 img = frame
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 faces = self.faceCascade.detectMultiScale(gray,1.3,5)
 print("Found {0} faces!".format(len(faces)))
 for (x, y, w, h) in faces:
 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
 img = frame[y:y + h, x:x + w]
 img = cv2.resize(img,(200,200))
31
 option = 1
 if option == 1:
 self.capture_img = img
 cv2.imshow("Capture Face",img)
 cv2.waitKey(0)
 else:
 messagebox.showinfo("Face or person not detected. Please try again","Face or person 
not detected. Please try again")
 def getClassLabel(self):
 path = 'student_details'
 student_size = []
 for root, dirs, directory in os.walk(path):
 name = os.path.basename(root)
 if name != path:
 student_size.append(name)
 size = len(student_size)
 return size
 def isUserExist(self,class_label):
 flag = 0
 for i in range(len(self.student_details)):
 arr = self.student_details[i].split(",")
 if arr[0] == str(class_label):
32
 flag = 1
 break
 return flag 
 
 def saveImage(self):
 class_label = self.getClassLabel()
 if self.capture_img is not None:
 sid = self.tf1.get()
 name = self.tf2.get()
 if os.path.exists('student_details/'+str(class_label)) == False:
 os.mkdir('student_details/'+str(class_label))
 if self.isUserExist(class_label) == 0:
 cv2.imwrite('student_details/'+str(class_label)+'/0.png',self.capture_img)
 cv2.imwrite('student_details/'+str(class_label)+'/1.png',self.capture_img)
 cv2.imwrite('student_details/'+str(class_label)+'/2.png',self.capture_img)
 cv2.imwrite('student_details/'+str(class_label)+'/3.png',self.capture_img)
 f = open("details.txt", "a+")
 f.write(str(class_label)+","+str(sid)+","+name+"\n")
 f.close()
 self.student_details.append(str(class_label)+","+str(sid)+","+name)
 return class_label 
 
 
 
33
 def trainmodel(self):
 size = self.saveImage()
 global labels
 global X_train
 global Y_train
 global classifier
 global prices
 labels = []
 X_train = []
 Y_train = []
 path = 'student_details'
 if size >= 1:
 for root, dirs, directory in os.walk(path):
 for j in range(len(directory)):
 name = os.path.basename(root)
 img = cv2.imread(root+"/"+directory[j])
 img = cv2.resize(img, (128,128))
 im2arr = np.array(img)
 im2arr = im2arr.reshape(128,128,3)
 X_train.append(im2arr)
 Y_train.append(int(name))
 X_train = np.asarray(X_train)
 Y_train = np.asarray(Y_train)
 print(Y_train)
34
 X_train = X_train.astype('float32')
 X_train = X_train/255
 indices = np.arange(X_train.shape[0])
 np.random.shuffle(indices)
 X_train = X_train[indices]
 Y_train = Y_train[indices]
 Y_train = to_categorical(Y_train)
 print(Y_train)
 classifier = Sequential()
 classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
 classifier.add(MaxPooling2D(pool_size = (2, 2)))
 classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
 classifier.add(MaxPooling2D(pool_size = (2, 2)))
 classifier.add(Flatten())
 classifier.add(Dense(output_dim = 256, activation = 'relu'))
 classifier.add(Dense(output_dim = Y_train.shape[1], activation = 'softmax'))
 print(classifier.summary())
 classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 
['accuracy'])
 hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True, 
verbose=2)
 acc = hist.history['accuracy']
 accuracy = acc[9] * 100
 messagebox.showinfo("Training model accuracy","Training Model Accuracy = 
35
"+str(accuracy))
 def predict(self):
 option = 0
 ret, frame = self.vid.get_frame()
 img = frame
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 faces = self.faceCascade.detectMultiScale(gray,1.3,5)
 print("Found {0} faces!".format(len(faces)))
 self.text.delete('1.0', END)
 for (x, y, w, h) in faces:
 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
 img = frame[y:y + h, x:x + w]
 img = cv2.resize(img,(128,128))
 option = 1
 if option == 1:
 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
 im2arr = np.array(img)
 im2arr = im2arr.reshape(1,128,128,3)
 image = np.asarray(im2arr)
 image = image.astype('float32')
 image = image/255
 preds = classifier.predict(image)
 predict = np.argmax(preds)
36
 print(predict)
 flag = 0
 for i in range(len(self.student_details)):
 arr = self.student_details[i].split(",")
 if arr[0] == str(predict):
 self.text.insert(END,"Student Recognized As : "+arr[2]+"\n")
 self.text.insert(END,"Attendance Collected for today's Date:\n")
 self.text.insert(END,str(date.today())+"\n\n")
 flag = 1
 break
 if flag == 0:
 messagebox.showinfo("Unable to recognized face","Unable to recognized face")
 else:
 messagebox.showinfo("Unable to recognized face","Unable to recognized face")
 
 def update(self):
 ret, frame = self.vid.get_frame()
 if ret:
 self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
 self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
 self.window.after(self.delay, self.update)
37
class MyVideoCapture:
 def __init__(self, video_source=0):
 
 self.vid = cv2.VideoCapture(video_source)
 if not self.vid.isOpened():
 raise ValueError("Unable to open video source", video_source)
 self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
 self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 self.pid = 0
 def get_frame(self):
 if self.vid.isOpened():
 ret, frame = self.vid.read()
 if ret:
 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 else:
 return (ret, None)
 else:
 return (ret, None)
 def __del__(self):
 if self.vid.isOpened():
 self.vid.release()
38
App(tkinter.Tk(), "Tkinter and OpenCV")