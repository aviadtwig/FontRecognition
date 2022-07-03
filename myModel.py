"""
Author: Aviad Twig
ID: 319090171

Semester: 2021a
Course: Intro for computer vision
Course No: 22928
"""

import os
import time
import cv2
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

# don't print keras warnings, info and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def make_train_set_and_val_set(train_file='train_set.h5', val_file='val_set.h5', wantVal=True):
    """
    Making the images train set and val set to numpy.array shape (N, 32, 32, 1).

    Making the labels train set and val set to categorical numpy.

    :param train_file: The h5 files that contain the images and the labels (as lists) of the
    train set, if not given train_file='train_set.h5'
    :param val_file: The h5 files that contain the images and the labels (as lists) of the
    validation set, if not given val_file='val_set.h5'
    :param wantVal: if we want to create validation set, if not given wantVal=True

    :return: x_train, y_train, x_val, y_val. if wantVal is False then x_val and y_val equal None
    """
    # Create the train set
    train_set = h5py.File(train_file, 'r')
    images = list(train_set['images'])
    labels = list(train_set['labels'])

    # Reshape all of the images by adding 1 in the end (to specified grayscale)
    for im in range(len(images)):
        images[im] = images[im].reshape(32, 32, 1)

    # Convert the images to np.array and the labels to categorical
    x_train = np.array(images)
    y_train = to_categorical(labels, num_classes=3)
    train_set.close()

    x_val, y_val = None, None
    if wantVal:
        # Create the val set
        val_set = h5py.File(val_file, 'r')
        images_val = list(val_set['images'])
        labels_val = list(val_set['labels'])

        # Reshape all of the images by adding 1 in the end (to specified grayscale)
        for i in range(len(images_val)):
            images_val[i] = images_val[i].reshape(32, 32, 1)

        # Convert the images to np.array and the labels to categorical
        x_val = np.array(images_val)
        y_val = to_categorical(labels_val, num_classes=3)
        val_set.close()

    return x_train, y_train, x_val, y_val


def build_x_test(db, im_names, start=0, stop=0):
    """
    Building the images set of the test set and making them numpy.array shape (N, 32, 32, 1).

    :param db: All of the data that is in the data set (from image name till wordBB).
    :param im_names: A list that contain all of the image name in the data set.
    :param start: The start point of the images from the data set, if not given equals 0
    :param stop: The stop point of the images from the data set, if not given equals 0

    :return: numpy.array shape (N, 32, 32, 1) that contain the images
    """
    # If stop not given then we create all of the data set
    if stop == 0:
        stop = len(im_names)

    # Initialization the list x == images
    x = []
    for i in range(start, stop):
        im = im_names[i]
        # The image and the chars bounding boxes in the image
        images = db['data'][im][:]
        charBB = db['data'][im].attrs['charBB']

        nC = charBB.shape[-1]
        for b_inx in range(nC):
            # The values of the bounding box, and cropping it by applying Perspective Transform Algorithm
            bb = charBB[:, :, b_inx]
            pts1 = np.float32([bb.T[0], bb.T[1], bb.T[3], bb.T[2]])
            pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            crop_img = cv2.warpPerspective(images, matrix, (400, 400))

            # Image in grayscale, resize to (32,32) and add to list
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            im_pil = Image.fromarray(gray)
            pil_crop = im_pil.resize((32, 32))
            np_image = np.array(pil_crop)
            x += [np_image]

    # Reshape all of the images by adding 1 in the end (to specified grayscale)
    for i in range(len(x)):
        x[i] = x[i].reshape(32, 32, 1)

    # Convert the images to np.array
    x_test = np.array(x)

    return x_test


def build_model_and_save(x_train, y_train, x_val=None, y_val=None):
    """
    Building the model using the train set and the val set we already built.
    Val set can be none but then the model will not have validation data.
    After the building the model saved in json file, and the weights saves in h5 file.
    After saving it will print model saved.

    :param x_train: A np.array (N,32,32,1) of the images in the train set
    :param y_train: A binary matrix representation of the classes of the train set
    :param x_val: A np.array (N,32,32,1) of the images in the val set, if not given x_val=None
    :param y_val: A binary matrix representation of the classes of the val set, if not given y_val=None

    :raise ValueError: if x_train or y_train are None
    """
    if x_train is None or y_train is None:
        raise ValueError("can't build a model you have to create x_train and y_train")

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    # Adam optimizers with learning rate of 0.0001
    op = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
    check_point = ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy',
                                  mode='max', save_best_only=True)
    # Printing summary of the model
    model.summary()

    start = time.time()
    # Fit the model with 25 epochs.
    # If x_val or y_val are None then without validation data
    if x_val is None or y_val is None:
        history = model.fit(x=x_train, y=y_train, epochs=25)
    else:
        history = model.fit(x=x_train, y=y_train, epochs=25,
                            validation_data=(x_val, y_val), callbacks=[earlyStopping, check_point])

    end = time.time()
    print("time model", (end - start), "sec")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model loss and accuracy')
    plt.ylabel('percentage')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss', 'accuracy', 'val_accuracy'], loc='upper left')
    plt.show()

    print("Saved model to disk")


def predict_set(loaded='best_model.h5', test_file='test.h5'):
    """
    Making prediction of the model, and save the prediction in h5 file.

    :param loaded: the h5 file that contain the model,
    if not given then loaded='best_model.h5'
    :param test_file: the h5 file that contain the data of the test on the model,
    if not given then test_file='test.h5'
    """

    # Building the np.array of images of the train set
    db = h5py.File(test_file, 'r')
    im_names = list(db['data'].keys())
    x_test = build_x_test(db, im_names)
    db.close()

    # load model
    model = load_model(loaded)
    # Make predict on the model and save the prediction to h5 file
    predict = model.predict(x_test)
    file = h5py.File("predict.h5", "w")
    file.create_dataset('predict', data=predict)
    file.close()
