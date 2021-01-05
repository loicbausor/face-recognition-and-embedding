# -*- coding: utf-8 -*-
"""
Data Generator module (done by us)
"""

import numpy as np
from tensorflow import keras
import cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(299,299), n_channels = 3,
             n_classes=203, shuffle=False, detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.detector = detector
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_lab_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_lab_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __preprocess_image(self, image_path, center = 127.5, scale = 128):
        width = self.dim[0]
        height = self.dim[1]

        # Loads the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop the face
        try: 
            face_x, face_y, face_width, face_height  = np.abs(self.detector.detectMultiScale(img)[0])
            crop_img = img[face_y:face_y+face_height, face_x:face_x+face_width]
            crop_img = cv2.resize(crop_img, (width,height))
        except: 
            crop_img = cv2.resize(img, (width,height))

        norm_image = (crop_img - center) / scale

        return norm_image


    def __data_generation(self, list_IDs_temp, list_lab_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=float)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i in range(len(list_IDs_temp)):
            # Store sample
            X[int(i),] =  self.__preprocess_image(list_IDs_temp[i])

            # Store class
            y[int(i)] = list_lab_temp[int(i)]

        ycat = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return [X,ycat],ycat





class DataGeneratorCenter(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(299,299), n_channels = 3,
             n_classes=203, shuffle=False, detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.detector = detector
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_lab_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_lab_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __preprocess_image(self, image_path, center = 127.5, scale = 128):
        width = self.dim[0]
        height = self.dim[1]

        # Loads the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop the face
        try: 
            face_x, face_y, face_width, face_height  = np.abs(self.detector.detectMultiScale(img)[0])
            crop_img = img[face_y:face_y+face_height, face_x:face_x+face_width]
            crop_img = cv2.resize(crop_img, (width,height))
        except: 
            crop_img = cv2.resize(img, (width,height))

        norm_image = (crop_img - center) / scale

        return norm_image


    def __data_generation(self, list_IDs_temp, list_lab_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=float)
        
        y = np.empty((self.batch_size), dtype=int)
        dummy = np.zeros(shape=(self.batch_size,1), dtype=float)

        # Generate data
        for i in range(len(list_IDs_temp)):
            # Store sample
            X[int(i),] =  self.__preprocess_image(list_IDs_temp[i])

            # Store class
            y[int(i)] = list_lab_temp[int(i)]
            
        ycat = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return (X,ycat),(ycat,dummy)
    
    
