"""
Author: Aviad Twig
ID: 319090171

Semester: 2021a
Course: Intro for computer vision
Course No: 22928
"""

import os
import train
import myModel
import result
from os import path
import warnings

# don't print keras warnings, info and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class bcolors:
    """
    colors and fonts for printed on console
    """
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def create_train_and_val(train_files=['SynthText.h5', 'train.h5'], val_file='SynthText_val.h5'):
    """

    This function try to create two h5 files, using the class train:

    1. train_set.h5 - that contain the train set divided by images and labels.
    2. val_set.h5 - that contain the validation set divided by images and labels.

    :param train_files: The h5 files that contain the data of train set, if not given then train_files=[
    'SynthText.h5', 'train.h5']
    :param val_file: The h5 file that contain the data of validation set, if not given
    then val_file='SynthText_val.h5'

    :raise FileExistsError: if train_files[0] is not in the project file, so we can not create train set

    :return: True is succeed
    """
    if train_files is None:
        raise TypeError('The file to train is None')
    elif type(train_files) != list:
        files_train = [train_files]
    else:
        files_train = train_files

    if val_file is None:
        warnings.warn('No validation will be create')
    elif type(val_file) != list:
        files_val = [val_file]
    else:
        files_val = val_file

    to_train = []
    to_val = []
    if path.exists(files_train[0]):
        to_train += [files_train[0]]
        for file in files_train:
            if path.exists(file):
                to_train += [file]
            else:
                warnings.warn('The file -', file, 'does not exists')
        train.build_train_or_val_set('train', to_train)

        for file in files_val:
            if path.exists(val_file):
                to_val += [file]
            else:
                warnings.warn('The file -', val_file, 'does not exists')
        train.build_train_or_val_set('val', to_val)
        return True

    raise FileExistsError('The file - ', files_train[0], 'does not exists')


def build_model(train_file='train_set.h5', val_file='val_set.h5'):
    """
    1. Making x_train and y_train from the file train_file.
    2. Making x_val and y_val from the file val_file.
    3. Building the model I have designed.
    4. Save the model on model.json and model.h5 (the model.h5 contains the weights).

    :param train_file: the h5 file that contain the train set, if not given then train_file='train_set.h5'
    :param val_file: the h5 file that contain the validation set, if not given then val_file='val_set.h5'

    :raise FileExistsError: if train_set is not in the project file and the function create_train_and_val raise error

    :return: True is succeed
    """
    x_train, y_train, x_val, y_val = None, None, None, None
    if path.exists(train_file):
        if path.exists(val_file):
            x_train, y_train, x_val, y_val = myModel.make_train_set_and_val_set(train_file=train_file,
                                                                                val_file=val_file)
        else:
            warnings.warn('no validation in the model')
    else:
        res = create_train_and_val()
        if res:
            if path.exists('val_set.h5'):
                x_train, y_train, x_val, y_val = myModel.make_train_set_and_val_set(train_file='train_set.h5',
                                                                                    val_file='val_set.h5')
            else:
                warnings.warn('no validation in the model')

    if x_val is None:
        myModel.build_model_and_save(x_train, y_train, x_val=None, y_val=None)
    else:
        myModel.build_model_and_save(x_train, y_train, x_val, y_val)

    return True


def predict_model(test_file='test.h5', loaded='best_model.h5'):
    """
    This function try to predict the model according to the test file, from the model that have been loaded


    :param test_file: the h5 file that contain the data of test set, if not given then test_file='test.h5'
    :param loaded: the h5 file that contain the model, if not given then loaded='model.h5'

    :raise FileExistsError: if loaded_model and loaded_weights are not in the project file and the build_model
    function raise error, or if test_file is not in the project file then we can not predict on an empty set

    :return: True is succeed
    """
    if not path.exists(test_file):
        raise FileExistsError('The file -', test_file, 'does not exists, can not create prediction')

    if path.exists(loaded):
        myModel.predict_set()
    else:
        res = build_model()
        if res:
            myModel.predict_set(test_file=test_file)

    return True


def result_models(test_file='test.h5', predict_file='predict.h5', just_create=False):
    """
    This function try to put out the result of the prediction to csv file

    :param test_file: The h5 file where the data of the test file was predict, if not given then test_file='test.h5'
    :param predict_file: The h5 file were the prediction have been made, if not given then predict_file='predict.h5'
    :param just_create:

    :raise FileExistsError: if test_file is not in the project files then we can't know what the set data,
    or if the predict_file is not in the project files and the function predict_model raise error

    :return: True is succeed
    """
    if not path.exists(test_file):
        raise FileExistsError('The file -', test_file, 'does not exists, can not create csv result')
    if not path.exists(predict_file):
        predict_model(test_file=test_file)

    if just_create:
        result.finish_file(test_file=test_file, predict_file=predict_file, accord_to='char', by_most=True)
    else:
        print("Does the test file contain fonts? (y/n)")
        withFonts = False

        s = input()
        if s == 'y' or s == 'Y':
            withFonts = True

        print("Please enter what you want to check (by the number):")
        print("1. Create csv only by predictions on char.")
        print("2. Create csv by prediction and relying on the fact that a word contains a single font.")
        print("3. Create csv by prediction on word.")
        print("4. All of the above.")
        print("If you put something else, the function will be over, and return False!")
        s = input()

        if s == '1':
            result.finish_file(test_file=test_file, predict_file=predict_file,
                               accord_to='char', by_most=False, withFonts=withFonts)
        elif s == '2':
            result.finish_file(test_file=test_file, predict_file=predict_file,
                               accord_to='char', by_most=True, withFonts=withFonts)
        elif s == '3':
            result.finish_file(test_file=test_file, predict_file=predict_file,
                               accord_to='word', withFonts=withFonts)
        elif s == '4':
            result.finish_file(test_file=test_file, predict_file=predict_file,
                               accord_to='char', by_most=False, withFonts=withFonts)
            result.finish_file(test_file=test_file, predict_file=predict_file,
                               accord_to='char', by_most=True, withFonts=withFonts)
            result.finish_file(test_file=test_file, predict_file=predict_file,
                               accord_to='word', withFonts=withFonts)
        else:
            return False

    return True


def questions(my_choice, had_prev=False):
    if my_choice == '4':
        print("\n" + bcolors.BLUE + "Do you have the file test.h5"
                                    " in your directory? (y/n)" + bcolors.ENDC)
        s = input()
        if s == 'y' or s == 'Y':
            print("\n" + bcolors.BLUE + "Do you have the file predict.h5"
                                        " in your directory? (y/n)" + bcolors.ENDC)
            s = input()
            if s == 'y' or s == 'Y':
                result_models(test_file='test.h5', predict_file='predict.h5')
            else:
                questions('3', had_prev=True)
                result_models(test_file='test.h5', predict_file='predict.h5')
        elif s == 'n' or s == 'N':
            print(bcolors.YELLOW + "Next time put the file test.h5 in your directory! " +
                  bcolors.ENDC + bcolors.RED + "BEY BEY!!" + bcolors.ENDC)
            exit(0)
    elif my_choice == '3':
        if not had_prev:
            print("\n" + bcolors.BLUE + "Do you have the file test.h5"
                                        " in your directory? (y/n)" + bcolors.ENDC)
            s = input()
            if s == 'n' or s == 'N':
                print(bcolors.YELLOW + "Next time put the file test.h5 in your directory! " +
                      bcolors.ENDC + bcolors.RED + "BEY BEY!!" + bcolors.ENDC)
                exit(0)
        print("\n" + bcolors.BLUE + "Do you have the loaded model 'best_model.h5'"
                                    " in your directory? (y/n)" + bcolors.ENDC)
        s = input()
        if s == 'y' or s == 'Y':
            predict_model(test_file='test.h5', loaded='best_model.h5')
        elif s == 'n' or s == 'N':
            questions('2', had_prev=True)
            predict_model(test_file='test.h5', loaded='best_model.h5')
    elif my_choice == '2':
        print("\n" + bcolors.BLUE + "Do you have the file train_set.h5 (and maybe val_set.h5)"
                                    " in your directory? (y/n)" + bcolors.ENDC)
        s = input()
        if s == 'y' or s == 'Y':
            build_model(train_file='train_set.h5', val_file='val_set.h5')
        elif s == 'n' or s == 'N':
            questions('1', had_prev=True)
            build_model(train_file='train_set.h5', val_file='val_set.h5')
    elif my_choice == '1':
        print("\n" + bcolors.BLUE + "Do you have the file SynthText.h5 in your directory? (y/n)" + bcolors.ENDC)
        s = input()
        if s == 'y' or s == 'Y':
            create_train_and_val(train_files=['SynthText.h5', 'train.h5'], val_file='SynthText_val.h5')
        elif s == 'n' or s == 'N':
            print("\n" + bcolors.YELLOW + "Next time put the file SynthText.h5 in your directory! " +
                  bcolors.ENDC + bcolors.RED + "BEY BEY!!" + bcolors.ENDC)
            exit(0)


print(bcolors.GREEN + bcolors.BOLD + "What would you like to check? specified the number" + bcolors.ENDC)
print("\n1. Predict the test from the model and create csv file.\n" +
      bcolors.UNDERLINE + bcolors.YELLOW +
      "Make sure you have in your directory the loaded model 'model.h5'\n"
      "and that you have the data of the data set 'test.h5'" + bcolors.ENDC)
print("\n2. Use other uses in the code.")
choice = input()
if choice == '1':
    predict_model(test_file='test.h5', loaded='best_model.h5')
    result_models(just_create=True)
elif choice == '2':
    print(bcolors.BLUE + bcolors.BOLD + "What would you like to check? specified the number" + bcolors.ENDC)
    print("1. Create the train set and validation set.")
    print("2. Build the model.")
    print("3. Predict from a model.")
    print("4. Get csv file of the results from the predictions.")
    print(bcolors.UNDERLINE + bcolors.YELLOW +
          "In every question if you put something else, the program will be over!\n\n" + bcolors.ENDC)

    first_choice = input()
    questions(first_choice)
