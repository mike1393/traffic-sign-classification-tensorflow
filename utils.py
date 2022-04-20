#Third Party
from cProfile import label
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
# Built-in
import os
import glob
import shutil
import csv
import errno
# from collections.abc import Sequence


def train_val_split(path_to_train: str, path_to_new_train: str, path_to_new_val: str, split_size: float=0.1):
    """Splits train data for validation

    Split the data into training data and validation data.
    Two new folders will be created containing new training data and new validation data repectively.

    Args:
        path_to_train: The path to training data contains GTSRB
        path_to_new_train: The path to new training data
        path_to_new_val: The path to new validation data
        split_size: The percentage of validation data with repect to old training data. Value between [0,1].

    Raise:
        FileNotFoundError: An error occurs if the path cannot be found by os

    """
    # Check whether the path can be found
    if not os.path.isdir(path_to_train):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_new_train)
    if not os.path.isdir(path_to_new_train):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_new_train)
    if not os.path.isdir(path_to_new_val):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_new_val)

    classification_folders = os.listdir(path_to_train)
    number_of_folders = len(classification_folders)
    print(f"{number_of_folders} class folder(s) found.")
    for idx, folder in enumerate(classification_folders):
        path_to_folder = os.path.join(path_to_train, folder)
        path_to_training_folder = os.path.join(path_to_new_train, folder)
        path_to_val_folder = os.path.join(path_to_new_val, folder)
        # Store all files within the folder into a list
        image_paths = glob.glob(os.path.join(path_to_folder, "*.png"))
        # Split the data into train and validate
        training_data, val_data = train_test_split(image_paths, test_size=split_size)
        # Copy the list of data to saperate folders
        print(f"Process folder :[{idx+1}/{number_of_folders}]", end=' ')
        copy_list_to_folder(training_data, path_to_training_folder, create=True)
        print(f"Process folder :[{idx+1}/{number_of_folders}]", end=' ')
        copy_list_to_folder(val_data, path_to_val_folder, create=True)
    print()


def copy_list_to_folder(files: list, path: str, create: bool=False):
    """Copies a list of files to target directory

    Each file will be copied to target directory.

    Args:
        files: A list of data path
        path: The target path
        create: A flag shows whether the user wants to create the path if not found.

    Raises:
        FileNotFoundError: An error occurs if file cannot be found by os

    """
    found = os.path.isdir(path)
    if not create and not found:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    if create and not found:
        create_if_not_found(path)
    number_of_files = len(files)
    for idx,file in enumerate(files):
        shutil.copy(file,path)
    print(f"Processed Data: {number_of_files}", end='\r')

def create_if_not_found(path: str):
    """ Create target path

    Recursively create target path if path is not found.

    Typical usage example:
    create_if_not_found("/foo")

    Args:
        path: the path to be found
    Return:
        None
    """
    found = os.path.isdir(path)
    if not found:
        os.makedirs(path)

def get_csv_row_number(csv_path: str)->int:
    """ Get the number of data from csv file

    Get the number of data from csv file by reading one row at a time.

    Typical usage example:
    number_of_data = get_csv_row_number("/data/Test.csv")
    
    Args:
        csv_path: The absolute path to test csv folder

    Return:
        The number of data in the csv file

    Raises:
        EnvironmentError: An error occurs if the csv.reader cannot open the file
    """
    try:
        data_size=0
        with open(csv_path,'rb') as file_path:
            reader = csv.reader(file_path)
            data_size = sum(1 for row in reader)
    except EnvironmentError:
        print(f"[Error] Cannot open file {csv_path}")
    return data_size

def order_test_set(path_to_test: str, path_to_test_csv: str):
    """ Organizes the testing images into label folders

    Re-organizes the images in test folder into cooresponding label.

    Typical usage example:
    order_test_set("/data/Test", "/data/Test.csv")

    Args:
        path_to_test: The absolute path to test folder
        path_to_test_csv: The absolute path to test csv folder
    Raises:
        EnvironmentError: if the csv.reader cannot open the file
    """
    image_size = get_csv_row_number(path_to_test_csv)-1
    print("Start ordering test files to labeled folder...\n")
    try:
        with open(path_to_test_csv, 'r') as csv_file:
            reader = csv.reader(csv_file,delimiter=",")
            for i,row in enumerate(reader):
                if i==0:
                    continue
                img_name = row[-1][5:]
                img_label = row[-2]
                path_to_folder = os.path.join(path_to_test,img_label)
                create_if_not_found(path_to_folder)
                path_to_image = os.path.join(path_to_test, img_name)
                shutil.move(path_to_image,path_to_folder)
                print(f"Processed Image :[{i}/{image_size}]", end='\r')
            print()
    except EnvironmentError:
        print(f"[Error] Cannot open file {path_to_test_csv}")

def create_generators(
    batch_size: int, path_to_train: str, path_to_val: str, 
    path_to_test: str, target_size=(40,40), class_mode: str='categorical'):
    """ Create image generator for training, validation, and testing data

    An ImageDataGenerator was created with a re-scaling factor of 1/255.
    Then the generator for each data path will be returned.

    Args:
        batch_size: The batch size for generator
        path_to_train: Full path to the training data folder
        path_to_val: Full path to the validation data folder
        path_to_test: Full path to the testing data folder
        target_size: Resize the image to target size
        class_mode: The encoded mode for generator, 
                    make sure to match the loss type during model compilation.
    Return:
        A tuple of (training_generator, validation_generator, testing_generator).
    """
    training_preprocessor = ImageDataGenerator(rescale=1/255., rotation_range=10, width_shift_range=0.1)
    test_preprocessor = ImageDataGenerator(rescale=1/255.)
    train_generator = training_preprocessor.flow_from_directory(
        directory=path_to_train,
        target_size=target_size,
        color_mode='rgb',
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True)

    val_generator = test_preprocessor.flow_from_directory(
        directory=path_to_val,
        target_size=target_size,
        color_mode='rgb',
        class_mode=class_mode,
        batch_size=1,
        shuffle=False)

    test_generator = test_preprocessor.flow_from_directory(
        directory=path_to_test,
        target_size=target_size,
        color_mode='rgb',
        class_mode=class_mode,
        batch_size=1,
        shuffle=False)

    return train_generator, val_generator, test_generator

def display_performance(fig_title, number_of_nets: int, epochs: int, history, names: list, ylim=[0.95,1]):
    # PLOT ACCURACIES
    plt.figure(figsize=(epochs+2,5))
    for i in range(number_of_nets):
        plt.plot(history[i].history['val_accuracy'],linestyle='-')
    title = f"{fig_title}\n" + "Model Accuracy Compare"
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper left')
    axes = plt.gca()
    axes.set_ylim(ylim)
    plt.show()

def display_history(history, epochs: int):

    # PLOT ACCURACIES
    names = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    fig = plt.figure(figsize=(epochs+2,5))
    fig.suptitle("Fitting Result")
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for i,name in enumerate(names[:]):
        if i<2:
            ax1.plot(history.history[name],linestyle='-',label=name)
        else:
            ax2.plot(history.history[name],linestyle='-', label=name)
    ax1.set_ylabel("accuracy")
    ax1.set_ylim([0.98,1])
    ax1.set_xlabel("epoch")
    ax2.set_ylim([0.0,0.1])
    ax2.set_ylabel("loss")
    ax2.set_xlabel("epoch")

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.subplots_adjust(hspace=0.35)
    plt.show()


def show_confusion_matrix(y_true, y_pred):
    
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(20, 20))
    # fig.suptitle("Confusion Matrix")
    ax = fig.add_subplot(111)
    ax.set_title("Confusion Matrix")
    sns.heatmap(confusion_mtx,
                annot=True, fmt='g',ax=ax, xticklabels=1, yticklabels=1)
    ylabel = [str(i) for i in range(43)]
    ax.set_yticklabels(ylabel, rotation =0)
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

def show_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    fig = plt.figure(figsize=(6, 70))
    # fig.suptitle("Classification Report")
    ax = fig.add_subplot(111)
    ax.set_title("Classification Report")
    clf_report = classification_report(y_true,
                                   y_pred,
                                   output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, 
                annot=True, xticklabels=1, yticklabels=1, ax=ax)

    ylabel = ['class'+str(i) for i in range(43)]
    ylabel+=['Accuracy', 'Macro Avg', 'Weighted Avg']
    ax.set_yticklabels(ylabel, rotation =0)
    
    plt.show()

def get_predict_from_generator(model, test_generator):
    labels = np.array([])
    imgs = np.array([])
    data_length = len(test_generator)
    for i in range(data_length):
        img, label = next(test_generator)
        if i==0:
            imgs = np.array(img)
            labels = np.array(label)
        else:
            imgs = np.vstack((imgs, img))
            labels = np.vstack((labels, label))
        print(f"[{i+1}/{data_length}]", end='\r')
    y_pred = np.argmax(model.predict(imgs,verbose=1), axis=1)
    y_label = np.argmax(labels, axis=1)
    return (y_label, y_pred)