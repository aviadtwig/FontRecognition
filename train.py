"""
Author: Aviad Twig
ID: 319090171

Semester: 2021a
Course: Intro for computer vision
Course No: 22928
"""

import os
import random
import cv2
import numpy as np
import h5py
from PIL import Image

# don't print keras warnings, info and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def noisy(image):
    """
    :param image: an image to add noise with salt and pepper
    :return: the noisy image
    """
    # Avoid aliasing
    blur = np.array(image)
    row, col = blur.shape

    # Add salt
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y and x coordinate
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        blur[y_coord][x_coord] = 255

    # Add pepper
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y and x coordinate
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        blur[y_coord][x_coord] = 0

    return blur


def images_and_labels_set(db, im_names, classes, start=0, stop=0, with_noise=True):
    """
    Create list of images and list of labels and adding noise to each image.
    This function is to create the training set or the validation set.

    :param db: All of the data that is in the data set (from image name till wordBB).
    :param im_names: A list that contain all of the image name in the data set.
    :param classes: A list of the categories in the set ['Skylark', 'Ubuntu Mono', 'Sweet Puppy'].
    :param start: The start point of the images from the data set, if not given equals 0
    :param stop: The stop point of the images from the data set, if not given equals 0
    :param with_noise: If True then it's the train set, if False it's the validation set. if not given with_noise=True

    :return: list of images (each image shape is (32,32))  and list of labels
    """
    # If stop not given then we create all of the data set
    if stop == 0:
        stop = len(im_names)

    # For train set each label enter twice (for the noisy image), and for the validation set only once
    mul_label = 1
    if with_noise:
        mul_label = 2

    # Initialization the lists. x == images, y == labels
    x = []
    y = []
    for i in range(start, stop):
        im = im_names[i]

        # The image, the fonts in the image, the chars bounding boxes in the image
        image = db['data'][im][:]
        font = db['data'][im].attrs['font']
        charBB = db['data'][im].attrs['charBB']
        nC = charBB.shape[-1]
        for b_inx in range(nC):
            # If mul=2 (meaning with noise) adding two labels one for the real image and the second to the noisy one
            # If mul=1 (meaning without noise) adding only one label
            if font[b_inx].decode('UTF-8') == classes[0]:
                y += [0] * mul_label
            elif font[b_inx].decode('UTF-8') == classes[1]:
                y += [1] * mul_label
            else:
                y += [2] * mul_label

            # The values of the bounding box, and cropping it by applying Perspective Transform Algorithm
            bb = charBB[:, :, b_inx]
            pts1 = np.float32([bb.T[0], bb.T[1], bb.T[3], bb.T[2]])
            pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            crop_img = cv2.warpPerspective(image, matrix, (400, 400))

            # Image in grayscale, resize to (32,32) and add to list
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            image_pil = Image.fromarray(gray)
            pil_crop = image_pil.resize((32, 32))
            np_crop_image = np.array(pil_crop)
            x += [np_crop_image]

            if with_noise:
                # Adding noise to the grayscale image, then resize to (32,32) and add to list
                crop_noisy_image = noisy(gray)
                noisy_pil = Image.fromarray(crop_noisy_image)
                pil_crop_noise = noisy_pil.resize((32, 32))
                np_crop_noise = np.array(pil_crop_noise)
                x += [np_crop_noise]

    return x, y


def store_h5_set(images, labels, toBe):
    """
    Stores an array of images and labels to HDF5.

    :param images: images array, (N, 32, 32, 1) to be stored
    :param labels: labels array, (N, 1) to be stored
    :param toBe: the type of the h5 file (train or val)
    """

    # Create a new HDF5 file
    file = h5py.File(toBe + "_set.h5", "w")

    # Create a dataset in the file
    file.create_dataset("images", np.shape(images), h5py.h5t.STD_U8BE, data=images)
    file.create_dataset("labels", np.shape(labels), h5py.h5t.STD_U8BE, data=labels)
    file.close()


def build_train_or_val_set(typeset, files):
    """
    Building the data set of the files that given, according to the type set (train or val)

    :param typeset: contain a string that defines the type of the set
    :param files: contain a list of the files name (h5 file) to build from them set

    :raise ValueError: if typeset isn't train or val
    """
    # If the files is not a list then we create a list
    if type(files) != list:
        files_name = [files]
    else:
        files_name = files

    # List to save the images and the labels
    x, y, = [], []
    # The categories
    font_name = ['Skylark', 'Ubuntu Mono', 'Sweet Puppy']

    for file_name in files_name:
        # Data and keys of the current file
        db = h5py.File(file_name, 'r')
        im_names = list(db['data'].keys())

        # Create imaged list and labels list of the current file according the typeset
        # If typeset is not train or val raise ValueError
        if typeset == 'train':
            x_temp, y_temp = images_and_labels_set(db, im_names, font_name)
        elif typeset == 'val':
            x_temp, y_temp = images_and_labels_set(db, im_names, font_name, with_noise=False)
        else:
            raise ValueError("Something went wrong, typeset unknown")

        # Add the current file lists to the return list
        x.extend(x_temp)
        y.extend(y_temp)

        # Close the file
        db.close()

    # Store the lists to h5 files
    store_h5_set(x, y, typeset)
