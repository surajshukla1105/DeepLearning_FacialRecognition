# -*- coding: utf-8 -*-
"""
Model Objectives: Facial detection and recognition to identify emotions
Models Used: CNN (Convolution Neural Network), VGG-16(Pretrained model)  
@author: Suraj Shukla
"""
import sys
vAR_vf = sys.argv[1]
#argument = 'C:/Users/Suraj Shukla/Desktop/TestImage1.jpg'
## import required Libraries for preprocessing
import numpy as np
from PIL import Image
import glob
import cv2 # import the opencv library
cv2.__version__ 
import os
import csv


## Enviornment settings
import configparser

configParser = configparser.RawConfigParser()   
vAR_configFilePath = 'C:/Users/Suraj Shukla/config.ini'
configParser.read(vAR_configFilePath)

vAR_PathPrj = configParser.get('My Section', 'vPathPrj')
vAR_PathResized  = configParser.get('My Section', 'vPathResized')
vAR_PathImg = configParser.get('My Section', 'vPathImg')
vAR_cascadePath = configParser.get('My Section', 'cascadePath')
vAR_ReadAllImages = configParser.get('My Section', 'vReadAllImages')
#vAR_vf = configParser.get('My Section', 'vf')

vAR_op = configParser.get('My Section', 'opt')

vAR_mw = configParser.get('My Section', 'mw')
vAR_ReadValidationImages  = configParser.get('My Section', 'vReadValidationImages')
vAR_TestImage  = configParser.get('My Section', 'vTestImage')
vAR_ValidationDataDirectory  = configParser.get('My Section', 'vValidationDataDirectory')
vAR_PathFinal = configParser.get('My Section', 'vPathFinal')
vAR_PathForTableau  = configParser.get('My Section', 'vPathForTableau')

## Read feature definition from config file
Emotion = configParser.items('Feature')
for i in range(0,5):
    Emotion[i] = str(Emotion[i])
    Emotion[i]= Emotion[i].split(',')[-1].split(')')[0]


vAR_PathPrj= vAR_PathPrj.strip() 


## Set DIrectory to project path
os.chdir(vAR_PathPrj)

os.getcwd()
## Detecting faces from webcam(Could be extended to use other cameras)
## I have used haarcascade_frontalface_default from open cv based on our requirement
## We can extend this as needed

detector= cv2.CascadeClassifier(vAR_cascadePath);

# Using LBPH                                
recognizer = cv2.face.createLBPHFaceRecognizer()

###################

## Setting up required variables
nb_epoch = 50
nb_train_samples = 670
nb_validation_samples = 84


#### Importing Keras (A wrapper on tensorflow for faster development of deep learning model)
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras import optimizers
from keras.utils import np_utils
import h5py

#from keras import backend as K
#K.set_image_vAR_dim_ordering('th')

batch_size = 4


os.chdir(vAR_PathPrj)
# dimensions of our images.
img_width, img_height = 150, 150
train_data_dir = vAR_PathResized
validation_data_dir = vAR_ValidationDataDirectory
datagen = ImageDataGenerator(rescale=1./255)
 # our data will be in order
# the predict_generator method returns the output of a model, given
# a generator that yields batches of numpy data



## Model Definition
model_vgg = Sequential()
model_vgg.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height,3)))
model_vgg.add(Conv2D(64, 3, 3, activation='relu', name='conv1_1'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(64, 3, 3, activation='relu', name='conv1_2'))
model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(128, 3, 3, activation='relu', name='conv2_1'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(128, 3, 3, activation='relu', name='conv2_2'))
model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(256, 3, 3, activation='relu', name='conv3_1'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(256, 3, 3, activation='relu', name='conv3_2'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(256, 3, 3, activation='relu', name='conv3_3'))
model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, 3, 3, activation='relu', name='conv4_1'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, 3, 3, activation='relu', name='conv4_2'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, 3, 3, activation='relu', name='conv4_3'))
model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, 3, 3, activation='relu', name='conv5_1'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, 3, 3, activation='relu', name='conv5_2'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, 3, 3, activation='relu', name='conv5_3'))
model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
#from keras import backend as K
#K.set_image_dim_ordering('th')

## Using Weights of VGG16 a pretrained model
f = h5py.File('vgg16_weights.h5')

for k in range(f.attrs['nb_layers']):
    if k >= len(model_vgg.layers) - 1:
        # we don't look at the last two layers in the savefile (fully-connected and activation)
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    layer = model_vgg.layers[k]

    if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

    layer.set_weights(weights)

f.close()



### Extracting features to train top layer 
## This is done so that we can use the pretrained model in our application
## just the top layer is changed to cater to our requirements


train_data = np.load(open('bottleneck_features_train.npy', 'rb'))


validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))


model_top = Sequential()
model_top.add(Flatten(input_shape=train_data.shape[1:]))
model_top.add(Dense(256, activation='relu'))
model_top.add(Dropout(0.8))
model_top.add(Dense(5, activation='softmax'))

#validation_labels = np_utils.to_categorical(validation_labels)
model_top.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_top.load_weights('bottleneck_40_epochs.h5')

#vAR_Evaluate  = model_top.evaluate(validation_data, validation_labels)

###########################################################################
cam = cv2.VideoCapture(vAR_vf)
vAR_dim = (150,150)
## Set up CSV connection to write data
f = open(vAR_op, "w", newline='')
c = csv.writer(f)
## Using HAAR Cascade to detect faces n images(This comes from OpenCV - an open source library for comuter vision)

faceCascade = cv2.CascadeClassifier(vAR_cascadePath);
pos_frame = cam.get(cv2.CAP_PROP_POS_FRAMES)
## Read image as frames, detect faces, store them and predict their classes.
while True:
    # Read data as frames
    ret, im =cam.read()
    
    if ret == True:
    #imgarray = im
    ## Converted image to Grayscale for analysis
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # Detect face
        facerect = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
            #facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=3, minSize=(3, 3))
        if len(facerect) > 0:
            print('face detected')
            color = (255, 255, 255)
            for (x,y,w,h) in facerect:
                # Draw a rectangle around face
                cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
    
                image = im[y: y + h, x: x + w]
                image = cv2.resize(image, vAR_dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(vAR_TestImage,image)
                ## Predict the class of image
                imgenerator = datagen.flow_from_directory(
                        vAR_PathFinal,
                        target_size=(img_width, img_height),
                        batch_size=16,
                        class_mode='categorical',
                        shuffle=False )
                Intermediate = model_vgg.predict_generator(imgenerator, 1)
                result = model_top.predict_classes(Intermediate)
                emotion = Emotion[int(result)]
                ## Put text on screen
                cv2.putText(im,emotion, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, color)
                pat = str(i)+'.jpg'
                PathImg = os.path.join(vAR_PathFinal, str(pat))
                pos_frame = cam.get(cv2.CAP_PROP_POS_FRAMES)
                ## Save image to file system
                cv2.imwrite(PathImg,image)
                
                ## Save data to CSV
                RowData = emotion,PathImg,i
                i = i+1
                c.writerow(RowData)
#        cv2.imshow('im',im) 
#                
#        if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT):
#        # If the number of captured frames is equal to the total number of frames,
#        # we stop
#            break
        cv2.imshow('im',im)
        if cv2.waitKey(10) and cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT):
            break
## Close connections    
#ret, im =cam.read()
cam.release()
cv2.destroyAllWindows()
f.close()