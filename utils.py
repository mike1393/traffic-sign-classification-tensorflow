#Third Party
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
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
        batch_size=batch_size,
        shuffle=False)

    test_generator = test_preprocessor.flow_from_directory(
        directory=path_to_test,
        target_size=target_size,
        color_mode='rgb',
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=False)

    return train_generator, val_generator, test_generator

def display_performance(number_of_nets: int, epochs: int, history, names: list, line_styles: list, ylim=[0.95,1]):
    # PLOT ACCURACIES
    plt.figure(figsize=(epochs+2,5))
    for i in range(number_of_nets):
        plt.plot(history[i].history['val_accuracy'],linestyle=line_styles[i])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper left')
    axes = plt.gca()
    axes.set_ylim(ylim)
    plt.show()