"""
Author: Aviad Twig
ID: 319090171

Semester: 2021a
Course: Intro for computer vision
Course No: 22928
"""

import os
import csv
import h5py

# don't print keras warnings, info and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def finish_file(test_file='test.h5', predict_file='predict.h5',
                accord_to='char', withFonts=False, by_most=True):
    """
    Creating the csv file the we had been asked.

    :param test_file: The h5 file that contain the data of the set we did predictions,
     if not given test_file='test.h5'
    :param predict_file: The h5 file that contain the predictions on the test_file set,
     if not given predict_file='predict.h5'
    :param withFonts: Defines whether the set contain font,
     if not given withFonts=False
    :param accord_to: Defines according to what we need to build the csv file of the prediction,
    according to char or to word. if not given accord_to='char'
    :param by_most: Defines whether the prediction rely on that word contain single font,
    if not given by_most=True
    """
    # load the data of the test file
    db_test = h5py.File(test_file, 'r')
    im_names = list(db_test['data'].keys())

    # load the data of the predictions
    db_predict = h5py.File(predict_file, 'r')
    prediction = list(db_predict['predict'])

    font_name = ['Skylark', 'Ubuntu Mono', 'Sweet Puppy']

    # specified which font we predict to each char
    predict_fonts = []
    for i in range(len(prediction)):
        im = list(prediction[i])
        predict_fonts += [im.index(max(im))]

    text_in_image = []
    real_fonts = []
    for j in range(len(im_names)):
        image_name = im_names[j]
        current_text = db_test['data'][image_name].attrs['txt']
        # saving in a list all of the the text in an image
        text_in_image += [[image_name, current_text]]
        # if the set have font saving all of the real font for each char
        if withFonts:
            real = db_test['data'][image_name].attrs['font']
            charBB = db_test['data'][image_name].attrs['charBB']
            nC = charBB.shape[-1]
            for b_inx in range(nC):
                # adding label of actual font
                if real[b_inx].decode('UTF-8') == font_name[0]:
                    real_fonts += [0]
                elif real[b_inx].decode('UTF-8') == font_name[1]:
                    real_fonts += [1]
                else:
                    real_fonts += [2]

    chars = []
    words = []
    for name, texts in text_in_image:
        for w in range(len(texts[:])):
            word = texts[:][w].decode('UTF-8')
            # saving in a list all of the words, were (words[0] = image_name and words[1] = current_word)
            words += [[name, word]]
            for ch in word:
                # saving in a list all of the chars, were (chars[0] = image_name and chars[1] = current_char)
                chars += [[name, ch]]

    # building the csv base on the form (accord_to and by_most)
    if accord_to == 'char' and not by_most:
        rows = build_rows_by_char(predict_fonts, chars, withFonts, real_fonts)
    elif accord_to == 'char' and by_most:
        rows = build_rows_by_char_by_most(predict_fonts, words, withFonts, real_fonts)
        accord_to = 'char_by_most'
    elif accord_to == 'word':
        rows = build_rows_by_words(predict_fonts, words, withFonts, real_fonts)

    # creating the csv file
    with open('predict_on_set_by_' + accord_to + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows[0])


def build_rows_by_char(predict_fonts, chars, withFonts, real_fonts):
    """
    This function create the rows for the csv file, by prediction of a char.
    It does not give importance that every word contains a single font.

    :param predict_fonts: A list that contain the font that predict for each char
    :param chars: A list of the chars in the set.
    For each i: chars[i][0] = image name, and chars[i][0] = the char.
    :param withFonts: Defines whether the set contain font.
    :param real_fonts: If the set contain fonts, then this list contain the real font for each char.
    Otherwise, this list is empty.

    :return: The rows for the csv file.
    """

    # The csv headers
    if withFonts:
        # If we have font in the set it will also contain column for the real font
        rows = [[['SN', 'image', 'char',
                  'Skylark', 'Sweet Puppy', 'Ubuntu Mono',
                  'real Skylark', 'real Sweet Puppy', 'real Ubuntu Mono']]]
    else:
        rows = [[['SN', 'image', 'char',
                  'Skylark', 'Sweet Puppy', 'Ubuntu Mono']]]

    # Loop on the size of the list of the predict fonts.
    # We can see that len(predict_fonts) equal to len(chars)
    for i in range(len(predict_fonts)):
        im_name = chars[i][0]
        char = chars[i][1]
        predict_font = predict_fonts[i]

        # Specified the column of the predict font for the current char (=row)
        skylark, ubuntu_mono, sweet_puppy = 0, 0, 0
        if predict_font == 0:
            skylark = 1
        elif predict_font == 1:
            ubuntu_mono = 1
        else:
            sweet_puppy = 1

        # If we have fonts, specified the column of the real font for the current char (=row)
        if withFonts:
            real_font = real_fonts[i]
            real_skylark, real_ubuntu_mono, real_sweet_puppy = 0, 0, 0
            if real_font == 0:
                real_skylark = 1
            elif real_font == 1:
                real_ubuntu_mono = 1
            else:
                real_sweet_puppy = 1

            # Adding the new row for the current char if we have font
            rows[0] += [[i, im_name, char, skylark, sweet_puppy, ubuntu_mono,
                         real_skylark, real_sweet_puppy, real_ubuntu_mono]]
        else:
            # Adding the new row for the current char if we don't have font
            rows[0] += [[i, im_name, char, skylark, sweet_puppy, ubuntu_mono]]
    return rows


def build_rows_by_char_by_most(predict_fonts, words, withFonts, real_fonts=None):
    """
    This function create the rows for the csv file, by prediction of a char.
    It does give importance that every word contains a single font.
    For each word the function counts how many predictions the word has from each font.
    And if there is indeed a font that appears most in the word,
     it is the font of all of the chars in the current word.

    :param predict_fonts: A list that contain the font that predict for each char
    :param words: A list of the words in the set.
    For each i: words[i][0] = image name, and words[i][0] = the word.
    :param withFonts: Defines whether the set contain font.
    :param real_fonts: If the set contain fonts, then this list contain the real font for each char.
    Otherwise, this list is empty.

    :return: The rows for the csv file.
    """
    # The csv headers
    if withFonts:
        # If we have font in the set it will also contain column for the real font
        rows = [[['SN', 'image', 'char',
                  'Skylark', 'Sweet Puppy', 'Ubuntu Mono',
                  'real Skylark', 'real Sweet Puppy', 'real Ubuntu Mono']]]
    else:
        rows = [[['SN', 'image', 'char',
                  'Skylark', 'Sweet Puppy', 'Ubuntu Mono']]]

    # Saving an index for a char
    index_char = 0

    # Loop on the size of the list of the words.
    # We can see that len(predict_fonts) equal to (for all i: sum(len(word[i][1])))
    for i in range(len(words)):
        # Define if there is indeed a font that appears most in the word.
        isMost = True
        im_name = words[i][0]
        current_word = words[i][1]

        # Counter for each font initialize in each loop to 0.
        count_skylark, count_ubuntu, count_puppy = 0, 0, 0

        # Saving temporary index
        temp_index = index_char

        # Loop on the current word
        for ch in range(len(current_word)):
            # The current char prediction
            predict_font = predict_fonts[temp_index]
            if predict_font == 0:
                count_skylark += 1
            elif predict_font == 1:
                count_ubuntu += 1
            else:
                count_puppy += 1

            temp_index += 1

        # Specified the column of the predict font for the current word (=len(word)*row)
        skylark, ubuntu_mono, sweet_puppy = 0, 0, 0
        # If skylark appears mostly
        if count_skylark > count_ubuntu and count_skylark > count_puppy:
            skylark = 1
        # If ubuntu mono appears mostly
        elif count_ubuntu > count_puppy and count_ubuntu > count_skylark:
            ubuntu_mono = 1
        # If sweet puppy appears mostly
        elif count_puppy > count_skylark and count_puppy > count_ubuntu:
            sweet_puppy = 1
        # If there is a tie for "first place"
        else:
            isMost = False

        # A second loop on the current word
        for ch in current_word:
            if not isMost:
                # If there was a tie for "first place", then each char indicate it own predictions
                predict_font = predict_fonts[index_char]
                skylark, ubuntu_mono, sweet_puppy = 0, 0, 0
                if predict_font == 0:
                    skylark = 1
                elif predict_font == 1:
                    ubuntu_mono = 1
                else:
                    sweet_puppy = 1

            if withFonts:
                # If we have fonts, specified the column of the real font for the current char (=row)
                real_font = real_fonts[index_char]
                real_skylark, real_ubuntu_mono, real_sweet_puppy = 0, 0, 0
                if real_font == 0:
                    real_skylark = 1
                elif real_font == 1:
                    real_ubuntu_mono = 1
                else:
                    real_sweet_puppy = 1

                # Adding the new row for the current char if we have font
                rows[0] += [[index_char, im_name, ch, skylark, sweet_puppy, ubuntu_mono,
                             real_skylark, real_sweet_puppy, real_ubuntu_mono]]
            else:
                # Adding the new row for the current char if we don't have font
                rows[0] += [[index_char, im_name, ch, skylark, sweet_puppy, ubuntu_mono]]
            # Increasing the index of the char
            index_char += 1
    return rows


def build_rows_by_words(predict_fonts, words, withFonts, real_fonts=None):
    """
    This function create the rows for the csv file, by prediction of a word.
    It does give importance that every word contains a single font.
    For each word the function counts how many predictions the word has from each font.
    And if there is indeed a font that appears most in the word it is the font of the current word.
    If there is a tie on the "first place" between two fonts, each font get half(=0.5) in there predictions.
    If there is a tie on the "first place" between three fonts, each font get third(=0.333) in there predictions.

    :param predict_fonts: A list that contain the font that predict for each char
    :param words: A list of the words in the set.
    For each i: words[i][0] = image name, and words[i][0] = the word.
    :param withFonts: Defines whether the set contain font.
    :param real_fonts: If the set contain fonts, then this list contain the real font for each char.
    Otherwise, this list is empty.

    :return: The rows for the csv file.
    """
    # The csv headers
    if withFonts:
        # If we have font in the set it will also contain column for the real font
        rows = [[['SN', 'image', 'word',
                  'Skylark', 'Sweet Puppy', 'Ubuntu Mono',
                  'real Skylark', 'real Sweet Puppy', 'real Ubuntu Mono']]]
    else:
        rows = [[['SN', 'image', 'word',
                  'Skylark', 'Sweet Puppy', 'Ubuntu Mono']]]

    # Saving an index for a char
    index_char = 0

    # Loop on the size of the list of the words.
    # We can see that len(predict_fonts) equal to (for all i: sum(len(word[i][1])))
    for i in range(len(words)):
        im_name = words[i][0]
        current_word = words[i][1]

        if withFonts:
            # If there are fonts in the set, whole of the word contain on font.
            # That equal to the font of the first char in that word
            real_font = real_fonts[index_char]
            # If we have fonts, specified the column of the real font for the current word (=row)
            real_skylark, real_ubuntu_mono, real_sweet_puppy = 0, 0, 0
            if real_font == 0:
                real_skylark = 1
            elif real_font == 1:
                real_ubuntu_mono = 1
            else:
                real_sweet_puppy = 1

        # Counter for each font initialize in each loop to 0.
        count_skylark, count_ubuntu, count_puppy = 0, 0, 0

        # Loop on the current word
        for ch in range(len(current_word)):
            # The current char prediction
            predict_font = predict_fonts[index_char]
            if predict_font == 0:
                count_skylark += 1
            elif predict_font == 1:
                count_ubuntu += 1
            else:
                count_puppy += 1
            # Increasing the index of the char
            index_char += 1

        # Specified the column of the predict font for the current word (=len(word)*row)
        skylark, ubuntu_mono, sweet_puppy = 0, 0, 0

        # If skylark appears mostly
        if count_skylark > count_ubuntu and count_skylark > count_puppy:
            skylark = 1
        # If ubuntu mono appears mostly
        elif count_ubuntu > count_puppy and count_ubuntu > count_skylark:
            ubuntu_mono = 1
        # If sweet puppy appears mostly
        elif count_puppy > count_skylark and count_puppy > count_ubuntu:
            sweet_puppy = 1
        # If there is a tie for "first place" between all three fonts
        elif count_skylark == count_ubuntu == count_puppy:
            skylark = 1 / 3.0
            ubuntu_mono = 1 / 3.0
            sweet_puppy = 1 / 3.0
        # If there is a tie for "first place" between only two fonts
        elif count_skylark == count_ubuntu:
            skylark = 0.5
            ubuntu_mono = 0.5
        elif count_ubuntu == count_puppy:
            ubuntu_mono = 0.5
            sweet_puppy = 0.5
        elif count_skylark == count_puppy:
            skylark = 0.5
            sweet_puppy = 0.5

        if withFonts:
            # Adding the new row for the current word if we have font
            rows[0] += [[i, im_name, current_word, skylark, sweet_puppy, ubuntu_mono,
                         real_skylark, real_sweet_puppy, real_ubuntu_mono]]
        else:
            # Adding the new row for the current char if we don't have font
            rows[0] += [[i, im_name, current_word, skylark, sweet_puppy, ubuntu_mono]]
    return rows
